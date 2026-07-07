using System.Runtime.InteropServices;

namespace AIBridge;

// ============================================================================
// P/Invoke 声明层
//
// 对应 crates/aibridge-ffi/include/aibridge.h 的全部 extern "C" 函数。
// 所有声明严格匹配头文件签名（opaque 指针 / char** / aibridge_bytes_t**）。
//
// 安全要点（设计文档第 7 节 + FFI 遗留问题）：
// 1. last_error 线程局部：调 FFI 失败后，必须在同一托管线程立即读取
//    aibridge_last_error() 转存为字符串，再抛异常。绝不可跨线程读取。
// 2. stream_next 串行：同一 stream 句柄不可并发 next（由 ChatStreamAsync 保证）。
// 3. aibridge_bytes_free / aibridge_string_free：必须调用，用 SafeHandle 兜底。
// 4. client / stream 句柄必须 destroy，用 SafeHandle / IDisposable 兜底。
// ============================================================================

/// <summary>
/// FFI 错误码常量（与 aibridge.h 的 #define 一一对应）。
/// </summary>
internal static class AibridgeStatus
{
    public const int Ok = 0;

    // stream_next 专用返回值
    public const int StreamChunk = 0; // 拉到一个 chunk
    public const int StreamEnd = 1;   // 流正常结束

    // 错误类别（负数）
    public const int Authentication = -1;
    public const int RateLimit = -2;
    public const int Validation = -3;
    public const int ModelNotFound = -4;
    public const int Api = -5;
    public const int Network = -6;
    public const int Timeout = -7;
    public const int UnsupportedCapability = -8;
    public const int ProviderNotFound = -9;
    public const int VoiceNotAvailable = -10;
    public const int ServiceUnavailable = -11;
    public const int Ffi = -100; // FFI 层通用错误（空指针、JSON 解析失败、panic）
}

/// <summary>
/// 二进制缓冲结构（对应 aibridge.h 的 aibridge_bytes_t）。
/// #[repr(C)] 保证 ptr + len 布局，C# 用 LayoutKind.Sequential 对齐。
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct AibridgeBytes
{
    public IntPtr ptr;  // const uint8_t*（Rust 分配）
    public UIntPtr len; // size_t
}

/// <summary>
/// Rust 分配的 C 字符串 SafeHandle。
/// 包装 aibridge_string_free，保证即使异常也会释放，避免内存泄漏。
/// </summary>
internal sealed class AibridgeStringHandle : SafeHandle
{
    public AibridgeStringHandle() : base(IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        // 传 nullptr 是安全的 no-op，直接释放
        Native.aibridge_string_free(handle);
        return true;
    }

    /// <summary>把句柄转为托管字符串（UTF-8 → string），并释放原生缓冲。</summary>
    public string? MarshalAndFree()
    {
        if (IsInvalid) return null;
        // 先拷贝为托管字符串，再释放原生内存（避免悬垂指针）
        string? s = Marshal.PtrToStringUTF8(handle);
        Dispose();
        return s;
    }
}

/// <summary>
/// Rust 分配的二进制缓冲 SafeHandle。
/// 包装 aibridge_bytes_free，保证二进制音频缓冲被释放。
/// </summary>
internal sealed class AibridgeBytesHandle : SafeHandle
{
    public AibridgeBytesHandle() : base(IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        Native.aibridge_bytes_free(handle);
        return true;
    }

    /// <summary>把缓冲拷贝为 byte[] 并释放原生内存。</summary>
    public byte[] MarshalAndFree()
    {
        if (IsInvalid) return Array.Empty<byte>();
        // 先解引用结构拿到 ptr + len；用 try/finally 保证异常路径也释放原生内存
        // （new byte[] 在超大 len 时可能 OOM，此时仍需释放 aibridge_bytes_t）
        byte[] data;
        try
        {
            AibridgeBytes b = Marshal.PtrToStructure<AibridgeBytes>(handle);
            // checked 防截断；len 超过 int.MaxValue 在 .NET byte[] 上限外，会抛异常但仍走 finally 释放
            int len = checked((int)b.len.ToUInt64());
            data = new byte[len];
            if (b.ptr != IntPtr.Zero && len > 0)
            {
                Marshal.Copy(b.ptr, data, 0, len);
            }
        }
        finally
        {
            Dispose(); // 释放原生 aibridge_bytes_t（含内部 [u8]）
        }
        return data;
    }
}

/// <summary>
/// P/Invoke 入口声明。DllImport 使用 "aibridge"（不带 lib 前缀和扩展名），
/// 运行时由 NativeLibrary 按 OS 解析为 libaibridge.{dylib,so} / aibridge.dll。
/// 动态库定位逻辑见 <see cref="NativeResolver"/>。
/// </summary>
internal static class Native
{
    private const string LibName = "aibridge";

    // 静态构造：注册动态库解析回调，支持 AIBRIDGE_LIB_PATH 环境变量与输出目录搜索。
    static Native()
    {
        NativeResolver.Register(LibName);
    }

    // —— 生命周期 —— ------------------------------------------------------------

    /// <summary>创建客户端（对应 aibridge_client_new）。</summary>
    /// <param name="provider">Provider 类型，UTF-8 C 字符串（如 "echo"、"openai"）。</param>
    /// <param name="configJson">ClientOptions 的 JSON，可为 null（默认配置）。</param>
    /// <returns>成功返 client 指针；失败返 IntPtr.Zero（读 last_error）。</returns>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr aibridge_client_new(byte[] provider, byte[]? configJson);

    /// <summary>启动客户端（对应 aibridge_client_start）。</summary>
    /// <returns>0 成功；负数为错误码。</returns>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_start(IntPtr client);

    /// <summary>释放客户端句柄（对应 aibridge_client_destroy，nullptr 安全）。</summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void aibridge_client_destroy(IntPtr client);

    // —— 阻塞式调用 —— ---------------------------------------------------------

    /// <summary>文本对话（对应 aibridge_client_chat）。</summary>
    /// <param name="requestJson">ChatRequest 的 JSON。</param>
    /// <param name="outResponseJson">输出 ChatCompletion 的 JSON，调用方 aibridge_string_free。</param>
    /// <returns>0 成功；负数为错误码。</returns>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_chat(
        IntPtr client,
        byte[] requestJson,
        ref IntPtr outResponseJson);

    /// <summary>文字转语音（对应 aibridge_client_speech，二进制走 aibridge_bytes_t）。</summary>
    /// <param name="requestJson">SpeechRequest 的 JSON。</param>
    /// <param name="outAudio">输出二进制音频缓冲（可能为 IntPtr.Zero）。</param>
    /// <param name="outMetaJson">输出 SpeechResult（不含 audio_data）的 JSON。</param>
    /// <returns>0 成功；负数为错误码。</returns>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_speech(
        IntPtr client,
        byte[] requestJson,
        ref IntPtr outAudio,
        ref IntPtr outMetaJson);

    // —— 流式 —— ----------------------------------------------------------------

    /// <summary>创建流式 stream 句柄（对应 aibridge_client_chat_stream）。</summary>
    /// <param name="requestJson">ChatRequest 的 JSON。</param>
    /// <param name="outStream">输出 stream 句柄。</param>
    /// <returns>0 成功；负数为错误码。</returns>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_chat_stream(
        IntPtr client,
        byte[] requestJson,
        ref IntPtr outStream);

    /// <summary>拉取下一个流式 chunk（阻塞，对应 aibridge_stream_next）。</summary>
    /// <returns>0=chunk，1=EOF，负数=错误。outChunkJson 为 chunk JSON（需 aibridge_string_free）。</returns>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_stream_next(IntPtr stream, ref IntPtr outChunkJson);

    /// <summary>释放 stream 句柄（触发 Rust drop → tokio task abort，nullptr 安全）。</summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void aibridge_stream_destroy(IntPtr stream);

    // —— 错误与释放 —— ----------------------------------------------------------

    /// <summary>读取当前线程的 last_error（JSON 字符串，调用方不应释放）。</summary>
    /// <returns>错误 JSON 指针；无错误返 IntPtr.Zero。线程局部，仅当前线程下一次 FFI 调用前有效。</returns>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr aibridge_last_error();

    /// <summary>释放 Rust 分配的 C 字符串（nullptr 安全）。</summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void aibridge_string_free(IntPtr ptr);

    /// <summary>释放 Rust 分配的二进制缓冲（nullptr 安全）。</summary>
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void aibridge_bytes_free(IntPtr ptr);
}

/// <summary>
/// 动态库运行时解析器。
///
/// .NET 默认只在系统库目录和输出目录搜索 libaibridge。为方便开发期直接
/// dotnet run（不打包），这里通过 NativeLibrary.SetDllImportResolver
/// 注入自定义解析：依次尝试
///   1) AIBRIDGE_LIB_PATH 环境变量指向的目录
///   2) 程序集输出目录（已通过 csproj CopyToOutputDirectory 拷贝过来）
///   3) 仓库 target/debug 与 target/release
/// 找到后用 NativeLibrary.Load 载入并缓存句柄。
/// </summary>
internal static class NativeResolver
{
    private static int _registered; // 0=未注册，1=已注册

    public static void Register(string libraryName)
    {
        if (Interlocked.CompareExchange(ref _registered, 1, 0) != 0) return;

        // 解析回调签名：(string libName, Assembly asm, DllImportSearchPath? searchPath, IntPtr) => IntPtr
        IntPtr Resolver(string lib, Assembly asm, DllImportSearchPath? search, IntPtr callers)
        {
            // 仅处理本绑定的库名（其它库走默认解析）
            if (!string.Equals(lib, libraryName, StringComparison.OrdinalIgnoreCase))
            {
                return IntPtr.Zero;
            }

            string?[] candidates =
            {
                Environment.GetEnvironmentVariable("AIBRIDGE_LIB_PATH"),
                AppContext.BaseDirectory,
            };

            // 候选目录优先级：环境变量 > 输出目录
            foreach (string? candidate in candidates)
            {
                if (string.IsNullOrEmpty(candidate)) continue;

                // 容错：candidate 可能是目录也可能是完整文件路径（用户误填）
                string dir = candidate;
                if (File.Exists(candidate))
                {
                    dir = Path.GetDirectoryName(candidate) ?? string.Empty;
                }

                if (string.IsNullOrEmpty(dir)) continue;
                string full = Path.Combine(dir, NativeFileName(libraryName));
                if (File.Exists(full))
                {
                    return NativeLibrary.Load(full, asm, DllImportSearchPath.Default);
                }
            }

            // 兜底：仓库 target/{debug,release}（相对输出目录回溯）
            string repoRoot = FindRepoRoot(AppContext.BaseDirectory);
            if (!string.IsNullOrEmpty(repoRoot))
            {
                foreach (string profile in new[] { "debug", "release" })
                {
                    string full = Path.Combine(repoRoot, "target", profile, NativeFileName(libraryName));
                    if (File.Exists(full))
                    {
                        return NativeLibrary.Load(full, asm, DllImportSearchPath.Default);
                    }
                }
            }

            // 最后让 .NET 用默认搜索（系统库目录等）
            return NativeLibrary.Load(libraryName, asm, DllImportSearchPath.Default);
        }

        NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, Resolver);
    }

    /// <summary>根据 OS 返回实际文件名（libaibridge.dylib / libaibridge.so / aibridge.dll）。</summary>
    private static string NativeFileName(string baseName)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return baseName + ".dll";
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            return "lib" + baseName + ".dylib";
        return "lib" + baseName + ".so"; // Linux/Unix
    }

    /// <summary>从起始目录向上查找仓库根（含 target/ + Cargo.toml 的目录）。</summary>
    private static string FindRepoRoot(string start)
    {
        DirectoryInfo? dir = new(start);
        while (dir != null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "target"))
                && File.Exists(Path.Combine(dir.FullName, "Cargo.toml")))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }
        return string.Empty;
    }
}
