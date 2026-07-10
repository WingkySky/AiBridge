using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

namespace AIBridge;

// ============================================================================
// Client 封装层
//
// 包装 aibridge_client_t* 句柄，提供 Chat / ChatStreamAsync / Speech 三个能力。
//
// 设计要点（FFI 遗留问题全部处理）：
// 1. last_error 线程局部：每次 FFI 失败后，在【同一托管线程】立即调
//    aibridge_last_error() 读取并转存为字符串，再交给 AibridgeException.FromStatus。
//    注意：因 ReadLastError 与 FFI 调用必须同线程，这里用普通同步方法实现核心逻辑，
//    再用 Task.Run 包装为异步暴露给上层（保证 FFI 调用 + last_error 读取在同一
//    ThreadPool 线程，不跨线程）。
// 2. stream_next 串行：ChatStreamAsync 的迭代器逐个 await，天然串行，不并发 next。
// 3. 字符串/字节释放：用 AibridgeStringHandle / AibridgeBytesHandle（SafeHandle）兜底。
// 4. 句柄释放：Client 实现 IDisposable，调 aibridge_client_destroy；Stream 同理。
// ============================================================================

/// <summary>
/// AIBridge 客户端。对应 core Client，通过 P/Invoke 调 aibridge-ffi。
/// </summary>
public sealed class Client : IDisposable
{
    private IntPtr _handle; // aibridge_client_t*
    private bool _started;
    private int _disposed; // 0=未释放，1=已释放（用 Interlocked 原子操作）

    // JSON 序列化选项：与 Rust serde 默认行为对齐。
    // 字段名已用 JsonPropertyName 显式指定 snake_case，不依赖命名策略转换。
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = null,
        // 容错：允许尾随逗号与注释（provider 响应可能含）
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    /// <summary>创建客户端（对应 aibridge_client_new）。</summary>
    /// <param name="provider">Provider 类型（如 "echo"、"openai"、"agnes"）。</param>
    /// <param name="configJson">ClientOptions 的 JSON，可为 null（默认配置）。</param>
    public Client(string provider, string? configJson = null)
    {
        if (string.IsNullOrEmpty(provider))
            throw new ArgumentException("provider 不能为空", nameof(provider));

        // C# string → UTF-8 字节数组（含 NUL 终止），匹配 C 字符串契约
        byte[] providerBytes = ToCString(provider);
        byte[]? configBytes = configJson != null ? ToCString(configJson) : null;

        IntPtr handle = Native.aibridge_client_new(providerBytes, configBytes);
        if (handle == IntPtr.Zero)
        {
            // 失败：同线程立即读 last_error（线程局部，不可跨线程）
            throw AibridgeException.FromStatus(AibridgeStatus.Ffi, ReadLastError());
        }
        _handle = handle;
    }

    /// <summary>启动客户端（对应 aibridge_client_start）。</summary>
    public void Start()
    {
        ThrowIfDisposed();
        if (_started) return;

        int status = Native.aibridge_client_start(_handle);
        if (status != AibridgeStatus.Ok)
        {
            throw AibridgeException.FromStatus(status, ReadLastError());
        }
        _started = true;
    }

    /// <summary>启动客户端（异步包装）。</summary>
    public Task StartAsync(CancellationToken cancellationToken = default)
        => Task.Run(Start, cancellationToken);

    // —— 文本对话 ——————————————————————————————————————————

    /// <summary>文本对话（阻塞，对应 aibridge_client_chat）。</summary>
    public ChatCompletion Chat(ChatRequest request)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        byte[] reqJson = ToCString(JsonSerializer.Serialize(request, JsonOpts));
        IntPtr outResponse = IntPtr.Zero;

        int status = Native.aibridge_client_chat(_handle, reqJson, ref outResponse);
        // 同线程立即读 last_error（线程局部，必须与 FFI 调用同线程）
        string? lastError = ReadLastError();

        if (status != AibridgeStatus.Ok)
        {
            // 失败时 outResponse 应为 Zero，但防御性释放
            if (outResponse != IntPtr.Zero) Native.aibridge_string_free(outResponse);
            throw AibridgeException.FromStatus(status, lastError);
        }

        // 成功：用 SafeHandle 接管字符串，拷贝后释放
        var handle = new AibridgeStringHandle(outResponse);
        string? responseJson = handle.MarshalAndFree();

        if (string.IsNullOrEmpty(responseJson))
        {
            throw new AibridgeException("chat 返回空响应");
        }

        return JsonSerializer.Deserialize<ChatCompletion>(responseJson, JsonOpts)
            ?? throw new AibridgeException("反序列化 ChatCompletion 失败");
    }

    /// <summary>文本对话（异步包装，用 Task.Run 在 ThreadPool 调度阻塞 FFI）。</summary>
    public Task<ChatCompletion> ChatAsync(ChatRequest request, CancellationToken cancellationToken = default)
        => Task.Run(() => Chat(request), cancellationToken);

    // —— 流式对话 ——————————————————————————————————————————

    /// <summary>
    /// 流式文本对话（对应 aibridge_client_chat_stream + stream_next 循环）。
    ///
    /// 返回 IAsyncEnumerable<ChatCompletionChunk>，可用 await foreach 消费。
    /// 内部 stream_next 串行调用（同一 stream 不可并发 next，符合 FFI 约束）。
    /// 取消通过 CancellationToken：迭代器在下次 next 前检查，并提前 destroy stream。
    /// </summary>
    public async IAsyncEnumerable<ChatCompletionChunk> ChatStreamAsync(
        ChatRequest request,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        byte[] reqJson = ToCString(JsonSerializer.Serialize(request, JsonOpts));
        IntPtr outStream = IntPtr.Zero;

        int status;
        string? lastError;
        // 创建 stream 必须同步完成（FFI 阻塞 + last_error 线程局部）
        status = Native.aibridge_client_chat_stream(_handle, reqJson, ref outStream);
        lastError = ReadLastError();

        if (status != AibridgeStatus.Ok)
        {
            throw AibridgeException.FromStatus(status, lastError);
        }

        // 用 try/finally 保证 stream 句柄一定被 destroy（即使中途取消或异常）
        try
        {
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();

                // stream_next 阻塞，包到 Task.Run 避免阻塞调用方线程。
                // 串行：每次 await 完成后才进入下一次 next，绝不并发。
                // 用元组返回结果，避免实例字段带来的并发隐患（同一 Client 可能有多个 stream）。
                (int nextStatus, string? chunkJson, string? lastError) result = await Task.Run(() =>
                {
                    IntPtr outChunk = IntPtr.Zero;
                    int s = Native.aibridge_stream_next(outStream, ref outChunk);
                    // 同线程读 last_error（线程局部，必须与 FFI 调用同线程）
                    string? err = s < 0 ? ReadLastError() : null;
                    string? json = null;
                    if (s == AibridgeStatus.StreamChunk && outChunk != IntPtr.Zero)
                    {
                        // 拷贝 chunk JSON 并释放原生字符串（SafeHandle 兜底释放）
                        var h = new AibridgeStringHandle(outChunk);
                        json = h.MarshalAndFree();
                    }
                    else if (outChunk != IntPtr.Zero)
                    {
                        // 防御性释放：FFI 在非 chunk 路径若意外写入 outChunk，避免泄漏
                        Native.aibridge_string_free(outChunk);
                    }
                    return (s, json, err);
                }, cancellationToken).ConfigureAwait(false);

                if (result.nextStatus == AibridgeStatus.StreamEnd)
                {
                    yield break; // 流正常结束
                }

                if (result.nextStatus < 0)
                {
                    throw AibridgeException.FromStatus(result.nextStatus, result.lastError);
                }

                // nextStatus == StreamChunk：反序列化 chunk
                if (string.IsNullOrEmpty(result.chunkJson))
                {
                    throw new AibridgeException("stream_next 返回 chunk 但 JSON 为空");
                }

                ChatCompletionChunk? chunk = JsonSerializer.Deserialize<ChatCompletionChunk>(result.chunkJson, JsonOpts);
                if (chunk == null)
                {
                    throw new AibridgeException("反序列化 ChatCompletionChunk 失败");
                }
                yield return chunk;
            }
        }
        finally
        {
            // 无论正常结束、取消、异常，都 destroy stream（触发 Rust drop → tokio task abort）
            if (outStream != IntPtr.Zero)
            {
                Native.aibridge_stream_destroy(outStream);
            }
        }
    }

    // —— 文字转语音 ——————————————————————————————————————————

    /// <summary>文字转语音（阻塞，对应 aibridge_client_speech，二进制走 aibridge_bytes_t）。</summary>
    public SpeechResult Speech(SpeechRequest request)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        byte[] reqJson = ToCString(JsonSerializer.Serialize(request, JsonOpts));
        IntPtr outAudio = IntPtr.Zero;
        IntPtr outMeta = IntPtr.Zero;

        int status = Native.aibridge_client_speech(_handle, reqJson, ref outAudio, ref outMeta);
        // 同线程立即读 last_error
        string? lastError = ReadLastError();

        if (status != AibridgeStatus.Ok)
        {
            // 防御性释放（FFI 失败时应为 Zero）
            if (outAudio != IntPtr.Zero) Native.aibridge_bytes_free(outAudio);
            if (outMeta != IntPtr.Zero) Native.aibridge_string_free(outMeta);
            throw AibridgeException.FromStatus(status, lastError);
        }

        // 二进制音频：用 SafeHandle 接管，拷贝后释放
        byte[] audioData = Array.Empty<byte>();
        if (outAudio != IntPtr.Zero)
        {
            var audioHandle = new AibridgeBytesHandle(outAudio);
            audioData = audioHandle.MarshalAndFree();
        }

        // 元数据 JSON
        SpeechResult? result;
        if (outMeta != IntPtr.Zero)
        {
            var metaHandle = new AibridgeStringHandle(outMeta);
            string? metaJson = metaHandle.MarshalAndFree();
            result = string.IsNullOrEmpty(metaJson)
                ? new SpeechResult()
                : JsonSerializer.Deserialize<SpeechResult>(metaJson, JsonOpts);
        }
        else
        {
            result = new SpeechResult();
        }

        result ??= new SpeechResult();
        result.AudioData = audioData;
        return result;
    }

    /// <summary>文字转语音（异步包装）。</summary>
    public Task<SpeechResult> SpeechAsync(SpeechRequest request, CancellationToken cancellationToken = default)
        => Task.Run(() => Speech(request), cancellationToken);

    // —— Dispose 模式 ——————————————————————————————————————————

    public void Dispose()
    {
        // 原子化：防并发 Dispose 导致 double-free aibridge_client_destroy
        if (Interlocked.Exchange(ref _disposed, 1) == 1) return;

        // 原子置换 _handle，确保只释放一次
        IntPtr h = Interlocked.Exchange(ref _handle, IntPtr.Zero);
        if (h != IntPtr.Zero)
        {
            Native.aibridge_client_destroy(h);
        }
    }

    // —— 内部辅助 ————————————————————————————————————————————

    /// <summary>读取当前线程的 last_error（线程局部，必须与 FFI 调用同线程）。</summary>
    private static string? ReadLastError()
    {
        IntPtr ptr = Native.aibridge_last_error();
        if (ptr == IntPtr.Zero) return null;
        // 拷贝为托管字符串（指针仅在当前线程下一次 FFI 调用前有效）
        return Marshal.PtrToStringUTF8(ptr);
    }

    /// <summary>C# string → UTF-8 字节数组（含 NUL 终止），匹配 C 字符串契约。</summary>
    private static byte[] ToCString(string s)
    {
        // Encoding.UTF8.GetBytes + 额外 NUL 字节
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        byte[] withNull = new byte[bytes.Length + 1];
        Buffer.BlockCopy(bytes, 0, withNull, 0, bytes.Length);
        return withNull;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(Client));
    }
}
