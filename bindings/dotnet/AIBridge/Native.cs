using System.Runtime.InteropServices;
using System.Reflection;

namespace AIBridge;

// ============================================================================
// P/Invoke 声明层
//
// 对应 crates/aibridge-ffi/include/aibridge.h 的全部 extern "C" 函数。
// 所有声明严格匹配头文件签名。
// ============================================================================

/// <summary>
/// FFI 错误码常量（与 aibridge.h 的 #define 一一对应）。
/// </summary>
internal static class AibridgeStatus
{
    public const int Ok = 0;
    public const int StreamChunk = 0;
    public const int StreamEnd = 1;
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
    public const int Ffi = -100;
}

/// <summary>
/// 二进制缓冲结构（对应 aibridge.h 的 aibridge_bytes_t）。
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct AibridgeBytes
{
    public IntPtr ptr;
    public UIntPtr len;
}

/// <summary>
/// Rust 分配的 C 字符串 SafeHandle。
/// </summary>
internal sealed class AibridgeStringHandle : SafeHandle
{
    public AibridgeStringHandle() : base(IntPtr.Zero, ownsHandle: true) { }
    public AibridgeStringHandle(IntPtr preexistingHandle) : base(preexistingHandle, ownsHandle: true) { }
    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        Native.aibridge_string_free(handle);
        return true;
    }

    public string? MarshalAndFree()
    {
        if (IsInvalid) return null;
        string? s = Marshal.PtrToStringUTF8(handle);
        Dispose();
        return s;
    }
}

/// <summary>
/// Rust 分配的二进制缓冲 SafeHandle。
/// </summary>
internal sealed class AibridgeBytesHandle : SafeHandle
{
    public AibridgeBytesHandle() : base(IntPtr.Zero, ownsHandle: true) { }
    public AibridgeBytesHandle(IntPtr preexistingHandle) : base(preexistingHandle, ownsHandle: true) { }
    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        Native.aibridge_bytes_free(handle);
        return true;
    }

    public byte[] MarshalAndFree()
    {
        if (IsInvalid) return Array.Empty<byte>();
        byte[] data;
        try
        {
            AibridgeBytes b = Marshal.PtrToStructure<AibridgeBytes>(handle);
            int len = checked((int)b.len.ToUInt64());
            data = new byte[len];
            if (b.ptr != IntPtr.Zero && len > 0)
                Marshal.Copy(b.ptr, data, 0, len);
        }
        finally { Dispose(); }
        return data;
    }
}

/// <summary>
/// P/Invoke 入口声明。
/// </summary>
internal static class Native
{
    private const string LibName = "aibridge";

    static Native()
    {
        NativeResolver.Register(LibName);
    }

    // —— 生命周期 ——

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr aibridge_client_new(byte[] provider, byte[]? configJson);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_start(IntPtr client);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void aibridge_client_destroy(IntPtr client);

    // —— 阻塞式调用 ——

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_chat(
        IntPtr client, byte[] requestJson, ref IntPtr outResponseJson);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_speech(
        IntPtr client, byte[] requestJson, ref IntPtr outAudio, ref IntPtr outMetaJson);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_image_generate(
        IntPtr client, byte[] requestJson, ref IntPtr outResponseJson);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_video_create(
        IntPtr client, byte[] requestJson, ref IntPtr outResponseJson);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_video_poll(
        IntPtr client, string taskId, string model, ref IntPtr outResponseJson);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_embed(
        IntPtr client, byte[] requestJson, ref IntPtr outResponseJson);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_transcribe(
        IntPtr client, byte[] requestJson, ref IntPtr outResponseJson);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_translate(
        IntPtr client, byte[] requestJson, ref IntPtr outResponseJson);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_list_models(
        IntPtr client, string filter, ref IntPtr outResponseJson);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_list_voices(
        IntPtr client, string language, ref IntPtr outResponseJson);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_recommend_voices(
        IntPtr client, string language, string gender, uint limit, ref IntPtr outResponseJson);

    // —— 流式 ——

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_client_chat_stream(
        IntPtr client, byte[] requestJson, ref IntPtr outStream);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int aibridge_stream_next(IntPtr stream, ref IntPtr outChunkJson);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void aibridge_stream_destroy(IntPtr stream);

    // —— 错误与释放 ——

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr aibridge_last_error();

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void aibridge_string_free(IntPtr ptr);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void aibridge_bytes_free(IntPtr ptr);
}

/// <summary>
/// 动态库运行时解析器。
/// </summary>
internal static class NativeResolver
{
    private static int _registered;

    public static void Register(string libraryName)
    {
        if (Interlocked.CompareExchange(ref _registered, 1, 0) != 0) return;

        IntPtr Resolver(string lib, Assembly asm, DllImportSearchPath? search)
        {
            if (!string.Equals(lib, libraryName, StringComparison.OrdinalIgnoreCase))
                return IntPtr.Zero;

            string?[] candidates =
            {
                Environment.GetEnvironmentVariable("AIBRIDGE_LIB_PATH"),
                AppContext.BaseDirectory,
            };

            foreach (string? candidate in candidates)
            {
                if (string.IsNullOrEmpty(candidate)) continue;

                string dir = candidate;
                if (File.Exists(candidate))
                    dir = Path.GetDirectoryName(candidate) ?? string.Empty;
                if (string.IsNullOrEmpty(dir)) continue;

                string full = Path.Combine(dir, NativeFileName(libraryName));
                if (File.Exists(full))
                    return NativeLibrary.Load(full, asm, null);
            }

            return NativeLibrary.Load(libraryName, asm, null);
        }

        NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, Resolver);
    }

    private static string NativeFileName(string baseName)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return baseName + ".dll";
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            return "lib" + baseName + ".dylib";
        return "lib" + baseName + ".so";
    }

    private static string FindRepoRoot(string start)
    {
        DirectoryInfo? dir = new(start);
        while (dir != null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "target"))
                && File.Exists(Path.Combine(dir.FullName, "Cargo.toml")))
                return dir.FullName;
            dir = dir.Parent;
        }
        return string.Empty;
    }
}
