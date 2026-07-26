using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AIBridge;

// ============================================================================
// Client 封装层
//
// 包装 aibridge_client_t* 句柄，提供全部能力。
// ============================================================================

public sealed class Client : IDisposable
{
    private IntPtr _handle;
    private bool _started;
    private int _disposed;

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = null,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    public Client(string provider, string? configJson = null)
    {
        if (string.IsNullOrEmpty(provider))
            throw new ArgumentException("provider 不能为空", nameof(provider));

        byte[] providerBytes = ToCString(provider);
        byte[]? configBytes = configJson != null ? ToCString(configJson) : null;

        IntPtr handle = Native.aibridge_client_new(providerBytes, configBytes);
        if (handle == IntPtr.Zero)
            throw AibridgeException.FromStatus(AibridgeStatus.Ffi, ReadLastError());
        _handle = handle;
    }

    public void Start()
    {
        ThrowIfDisposed();
        if (_started) return;
        int status = Native.aibridge_client_start(_handle);
        if (status != AibridgeStatus.Ok)
            throw AibridgeException.FromStatus(status, ReadLastError());
        _started = true;
    }

    public Task StartAsync(CancellationToken cancellationToken = default)
        => Task.Run(Start, cancellationToken);

    // ────────────────────────────────────────────────────────────────────
    // 文本对话
    // ────────────────────────────────────────────────────────────────────

    public ChatCompletion Chat(ChatRequest request)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        byte[] reqJson = ToCString(JsonSerializer.Serialize(request, JsonOpts));
        IntPtr outResponse = IntPtr.Zero;

        int status = Native.aibridge_client_chat(_handle, reqJson, ref outResponse);
        string? lastError = ReadLastError();

        if (status != AibridgeStatus.Ok)
        {
            if (outResponse != IntPtr.Zero) Native.aibridge_string_free(outResponse);
            throw AibridgeException.FromStatus(status, lastError);
        }

        var handle_ = new AibridgeStringHandle(outResponse);
        string? responseJson = handle_.MarshalAndFree();
        if (string.IsNullOrEmpty(responseJson))
            throw new AibridgeException("chat 返回空响应");

        return JsonSerializer.Deserialize<ChatCompletion>(responseJson, JsonOpts)
            ?? throw new AibridgeException("反序列化 ChatCompletion 失败");
    }

    public Task<ChatCompletion> ChatAsync(ChatRequest request, CancellationToken cancellationToken = default)
        => Task.Run(() => Chat(request), cancellationToken);

    public async IAsyncEnumerable<ChatCompletionChunk> ChatStreamAsync(
        ChatRequest request,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        byte[] reqJson = ToCString(JsonSerializer.Serialize(request, JsonOpts));
        IntPtr outStream = IntPtr.Zero;

        int status = Native.aibridge_client_chat_stream(_handle, reqJson, ref outStream);
        string? lastError = ReadLastError();

        if (status != AibridgeStatus.Ok)
            throw AibridgeException.FromStatus(status, lastError);

        try
        {
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();
                (int nextStatus, string? chunkJson, string? lastErr) result = await Task.Run(() =>
                {
                    IntPtr outChunk = IntPtr.Zero;
                    int s = Native.aibridge_stream_next(outStream, ref outChunk);
                    string? err = s < 0 ? ReadLastError() : null;
                    string? json = null;
                    if (s == AibridgeStatus.StreamChunk && outChunk != IntPtr.Zero)
                    {
                        var h = new AibridgeStringHandle(outChunk);
                        json = h.MarshalAndFree();
                    }
                    else if (outChunk != IntPtr.Zero)
                    {
                        Native.aibridge_string_free(outChunk);
                    }
                    return (s, json, err);
                }, cancellationToken).ConfigureAwait(false);

                if (result.nextStatus == AibridgeStatus.StreamEnd) yield break;
                if (result.nextStatus < 0)
                    throw AibridgeException.FromStatus(result.nextStatus, result.lastErr);

                ChatCompletionChunk? chunk = JsonSerializer.Deserialize<ChatCompletionChunk>(result.chunkJson, JsonOpts);
                if (chunk == null)
                    throw new AibridgeException("反序列化 ChatCompletionChunk 失败");
                yield return chunk;
            }
        }
        finally
        {
            if (outStream != IntPtr.Zero)
                Native.aibridge_stream_destroy(outStream);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // 文字转语音
    // ────────────────────────────────────────────────────────────────────

    public SpeechResult Speech(SpeechRequest request)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        byte[] reqJson = ToCString(JsonSerializer.Serialize(request, JsonOpts));
        IntPtr outAudio = IntPtr.Zero;
        IntPtr outMeta = IntPtr.Zero;

        int status = Native.aibridge_client_speech(_handle, reqJson, ref outAudio, ref outMeta);
        string? lastError = ReadLastError();

        if (status != AibridgeStatus.Ok)
        {
            if (outAudio != IntPtr.Zero) Native.aibridge_bytes_free(outAudio);
            if (outMeta != IntPtr.Zero) Native.aibridge_string_free(outMeta);
            throw AibridgeException.FromStatus(status, lastError);
        }

        byte[] audioData = Array.Empty<byte>();
        if (outAudio != IntPtr.Zero)
        {
            var audioHandle = new AibridgeBytesHandle(outAudio);
            audioData = audioHandle.MarshalAndFree();
        }

        SpeechResult? result;
        if (outMeta != IntPtr.Zero)
        {
            var metaHandle = new AibridgeStringHandle(outMeta);
            string? metaJson = metaHandle.MarshalAndFree();
            result = string.IsNullOrEmpty(metaJson) ? new SpeechResult()
                : JsonSerializer.Deserialize<SpeechResult>(metaJson, JsonOpts);
        }
        else { result = new SpeechResult(); }

        result ??= new SpeechResult();
        result.AudioData = audioData;
        return result;
    }

    public Task<SpeechResult> SpeechAsync(SpeechRequest request, CancellationToken cancellationToken = default)
        => Task.Run(() => Speech(request), cancellationToken);

    // ────────────────────────────────────────────────────────────────────
    // 图像生成
    // ────────────────────────────────────────────────────────────────────

    public ImageResult ImageGenerate(ImageRequest request)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        byte[] reqJson = ToCString(JsonSerializer.Serialize(request, JsonOpts));
        IntPtr outResponse = IntPtr.Zero;

        int status = Native.aibridge_client_image_generate(_handle, reqJson, ref outResponse);
        string? lastError = ReadLastError();

        if (status != AibridgeStatus.Ok)
        {
            if (outResponse != IntPtr.Zero) Native.aibridge_string_free(outResponse);
            throw AibridgeException.FromStatus(status, lastError);
        }

        var handle_ = new AibridgeStringHandle(outResponse);
        string? responseJson = handle_.MarshalAndFree();
        return JsonSerializer.Deserialize<ImageResult>(responseJson, JsonOpts)
            ?? throw new AibridgeException("反序列化 ImageResult 失败");
    }

    public Task<ImageResult> ImageGenerateAsync(ImageRequest request, CancellationToken cancellationToken = default)
        => Task.Run(() => ImageGenerate(request), cancellationToken);

    // ────────────────────────────────────────────────────────────────────
    // 视频生成
    // ────────────────────────────────────────────────────────────────────

    public VideoTask VideoCreate(VideoRequest request)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        byte[] reqJson = ToCString(JsonSerializer.Serialize(request, JsonOpts));
        IntPtr outResponse = IntPtr.Zero;

        int status = Native.aibridge_client_video_create(_handle, reqJson, ref outResponse);
        string? lastError = ReadLastError();

        if (status != AibridgeStatus.Ok)
        {
            if (outResponse != IntPtr.Zero) Native.aibridge_string_free(outResponse);
            throw AibridgeException.FromStatus(status, lastError);
        }

        var handle_ = new AibridgeStringHandle(outResponse);
        string? responseJson = handle_.MarshalAndFree();
        return JsonSerializer.Deserialize<VideoTask>(responseJson, JsonOpts)
            ?? throw new AibridgeException("反序列化 VideoTask 失败");
    }

    public Task<VideoTask> VideoCreateAsync(VideoRequest request, CancellationToken cancellationToken = default)
        => Task.Run(() => VideoCreate(request), cancellationToken);

    public VideoStatus VideoPoll(string taskId, string model)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(taskId);
        ArgumentNullException.ThrowIfNull(model);

        IntPtr outResponse = IntPtr.Zero;
        int status = Native.aibridge_client_video_poll(_handle, taskId, model, ref outResponse);
        string? lastError = ReadLastError();

        if (status != AibridgeStatus.Ok)
        {
            if (outResponse != IntPtr.Zero) Native.aibridge_string_free(outResponse);
            throw AibridgeException.FromStatus(status, lastError);
        }

        var handle_ = new AibridgeStringHandle(outResponse);
        string? responseJson = handle_.MarshalAndFree();
        return JsonSerializer.Deserialize<VideoStatus>(responseJson, JsonOpts)
            ?? throw new AibridgeException("反序列化 VideoStatus 失败");
    }

    public Task<VideoStatus> VideoPollAsync(string taskId, string model, CancellationToken cancellationToken = default)
        => Task.Run(() => VideoPoll(taskId, model), cancellationToken);

    // ────────────────────────────────────────────────────────────────────
    // 文本嵌入
    // ────────────────────────────────────────────────────────────────────

    public EmbeddingResult Embed(EmbedRequest request)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        byte[] reqJson = ToCString(JsonSerializer.Serialize(request, JsonOpts));
        IntPtr outResponse = IntPtr.Zero;

        int status = Native.aibridge_client_embed(_handle, reqJson, ref outResponse);
        string? lastError = ReadLastError();

        if (status != AibridgeStatus.Ok)
        {
            if (outResponse != IntPtr.Zero) Native.aibridge_string_free(outResponse);
            throw AibridgeException.FromStatus(status, lastError);
        }

        var handle_ = new AibridgeStringHandle(outResponse);
        string? responseJson = handle_.MarshalAndFree();
        return JsonSerializer.Deserialize<EmbeddingResult>(responseJson, JsonOpts)
            ?? throw new AibridgeException("反序列化 EmbeddingResult 失败");
    }

    public Task<EmbeddingResult> EmbedAsync(EmbedRequest request, CancellationToken cancellationToken = default)
        => Task.Run(() => Embed(request), cancellationToken);

    // ────────────────────────────────────────────────────────────────────
    // 语音转文字 / 翻译
    // ────────────────────────────────────────────────────────────────────

    public TranscriptionResult Transcribe(TranscribeRequest request)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        byte[] reqJson = ToCString(JsonSerializer.Serialize(request, JsonOpts));
        IntPtr outResponse = IntPtr.Zero;

        int status = Native.aibridge_client_transcribe(_handle, reqJson, ref outResponse);
        string? lastError = ReadLastError();

        if (status != AibridgeStatus.Ok)
        {
            if (outResponse != IntPtr.Zero) Native.aibridge_string_free(outResponse);
            throw AibridgeException.FromStatus(status, lastError);
        }

        var handle_ = new AibridgeStringHandle(outResponse);
        string? responseJson = handle_.MarshalAndFree();
        return JsonSerializer.Deserialize<TranscriptionResult>(responseJson, JsonOpts)
            ?? throw new AibridgeException("反序列化 TranscriptionResult 失败");
    }

    public Task<TranscriptionResult> TranscribeAsync(TranscribeRequest request, CancellationToken cancellationToken = default)
        => Task.Run(() => Transcribe(request), cancellationToken);

    public TranscriptionResult Translate(TranscribeRequest request)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        byte[] reqJson = ToCString(JsonSerializer.Serialize(request, JsonOpts));
        IntPtr outResponse = IntPtr.Zero;

        int status = Native.aibridge_client_translate(_handle, reqJson, ref outResponse);
        string? lastError = ReadLastError();

        if (status != AibridgeStatus.Ok)
        {
            if (outResponse != IntPtr.Zero) Native.aibridge_string_free(outResponse);
            throw AibridgeException.FromStatus(status, lastError);
        }

        var handle_ = new AibridgeStringHandle(outResponse);
        string? responseJson = handle_.MarshalAndFree();
        return JsonSerializer.Deserialize<TranscriptionResult>(responseJson, JsonOpts)
            ?? throw new AibridgeException("反序列化 TranscriptionResult 失败");
    }

    public Task<TranscriptionResult> TranslateAsync(TranscribeRequest request, CancellationToken cancellationToken = default)
        => Task.Run(() => Translate(request), cancellationToken);

    // ────────────────────────────────────────────────────────────────────
    // 模型列表 / 音色列表
    // ────────────────────────────────────────────────────────────────────

    public List<ModelInfo> ListModels(string? filter = null)
    {
        ThrowIfDisposed();

        IntPtr outResponse = IntPtr.Zero;
        int status = Native.aibridge_client_list_models(_handle, filter ?? "", ref outResponse);
        string? lastError = ReadLastError();

        if (status != AibridgeStatus.Ok)
        {
            if (outResponse != IntPtr.Zero) Native.aibridge_string_free(outResponse);
            throw AibridgeException.FromStatus(status, lastError);
        }

        var handle_ = new AibridgeStringHandle(outResponse);
        string? responseJson = handle_.MarshalAndFree();
        return JsonSerializer.Deserialize<List<ModelInfo>>(responseJson, JsonOpts)
            ?? throw new AibridgeException("反序列化 Vec<ModelInfo> 失败");
    }

    public Task<List<ModelInfo>> ListModelsAsync(string? filter = null, CancellationToken cancellationToken = default)
        => Task.Run(() => ListModels(filter), cancellationToken);

    public List<VoiceInfo> ListVoices(string? language = null)
    {
        ThrowIfDisposed();

        IntPtr outResponse = IntPtr.Zero;
        int status = Native.aibridge_client_list_voices(_handle, language ?? "", ref outResponse);
        string? lastError = ReadLastError();

        if (status != AibridgeStatus.Ok)
        {
            if (outResponse != IntPtr.Zero) Native.aibridge_string_free(outResponse);
            throw AibridgeException.FromStatus(status, lastError);
        }

        var handle_ = new AibridgeStringHandle(outResponse);
        string? responseJson = handle_.MarshalAndFree();
        return JsonSerializer.Deserialize<List<VoiceInfo>>(responseJson, JsonOpts)
            ?? throw new AibridgeException("反序列化 Vec<VoiceInfo> 失败");
    }

    public Task<List<VoiceInfo>> ListVoicesAsync(string? language = null, CancellationToken cancellationToken = default)
        => Task.Run(() => ListVoices(language), cancellationToken);

    public List<VoiceInfo> RecommendVoices(string? language = null, string? gender = null, uint limit = 10)
    {
        ThrowIfDisposed();

        IntPtr outResponse = IntPtr.Zero;
        int status = Native.aibridge_client_recommend_voices(
            _handle, language ?? "", gender ?? "", limit, ref outResponse);
        string? lastError = ReadLastError();

        if (status != AibridgeStatus.Ok)
        {
            if (outResponse != IntPtr.Zero) Native.aibridge_string_free(outResponse);
            throw AibridgeException.FromStatus(status, lastError);
        }

        var handle_ = new AibridgeStringHandle(outResponse);
        string? responseJson = handle_.MarshalAndFree();
        return JsonSerializer.Deserialize<List<VoiceInfo>>(responseJson, JsonOpts)
            ?? throw new AibridgeException("反序列化 Vec<VoiceInfo> 失败");
    }

    public Task<List<VoiceInfo>> RecommendVoicesAsync(string? language = null, string? gender = null,
        uint limit = 10, CancellationToken cancellationToken = default)
        => Task.Run(() => RecommendVoices(language, gender, limit), cancellationToken);

    // ────────────────────────────────────────────────────────────────────
    // Dispose
    // ────────────────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 1) return;
        IntPtr h = Interlocked.Exchange(ref _handle, IntPtr.Zero);
        if (h != IntPtr.Zero)
            Native.aibridge_client_destroy(h);
    }

    private static string? ReadLastError()
    {
        IntPtr ptr = Native.aibridge_last_error();
        if (ptr == IntPtr.Zero) return null;
        return Marshal.PtrToStringUTF8(ptr);
    }

    private static byte[] ToCString(string s)
    {
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
