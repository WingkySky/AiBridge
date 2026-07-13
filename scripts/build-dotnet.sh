#!/usr/bin/env bash
# build-dotnet.sh - 构建 .NET 绑定（dotnet pack，含动态库打进 NuGet）
#
# 产物：bindings/dotnet/AIBridge/bin/Release/*.nupkg
# 动态库放进 runtimes/{rid}/native/（.NET 标准，运行时 NativeLibrary 自动加载）。
# 若本机无 dotnet SDK，打印提示并退出 0（不阻塞 release.sh）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v dotnet >/dev/null 2>&1; then
  echo "==> 未检测到 dotnet SDK，跳过 .NET 构建"
  echo "    安装：https://dotnet.microsoft.com/download"
  echo "    macOS: brew install --cask dotnet-sdk"
  exit 0
fi

echo "==> 先构建 libaibridge (release)"
"$REPO_ROOT/scripts/build-ffi.sh"

# 计算 .NET RID + 动态库文件名
case "$(uname -s)" in
  Darwin)
    case "$(uname -m)" in
      arm64|aarch64) RID="osx-arm64" ;;
      *)             RID="osx-x64" ;;
    esac
    LIB="libaibridge.dylib" ;;
  Linux)
    case "$(uname -m)" in
      aarch64|arm64) RID="linux-arm64" ;;
      *)             RID="linux-x64" ;;
    esac
    LIB="libaibridge.so" ;;
  MINGW*|MSYS*|CYGWIN*)
    RID="win-x64"
    LIB="aibridge.dll" ;;
  *)
    echo "错误：不支持的操作系统 $(uname -s)" >&2
    exit 1 ;;
esac

LIBDIR="$REPO_ROOT/target/release"
NATIVE_DIR="$REPO_ROOT/bindings/dotnet/AIBridge/runtimes/$RID/native"
mkdir -p "$NATIVE_DIR"
cp "$LIBDIR/$LIB" "$NATIVE_DIR/"
echo "==> 拷贝 $LIB -> runtimes/$RID/native/"

cd "$REPO_ROOT/bindings/dotnet/AIBridge"
echo "==> dotnet pack（Release）"
dotnet pack -c Release
echo "==> NuGet 产物:"
ls -lh bin/Release/*.nupkg 2>/dev/null || true
