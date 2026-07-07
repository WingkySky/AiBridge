#!/usr/bin/env bash
# build-ffi.sh - 构建 aibridge-ffi 动态库（libaibridge.{so,dylib,dll}）
#
# 产物供 Go / JVM / .NET 绑定消费。
# 默认 release 模式（产物在 target/release/）；可用 PROFILE=debug 切换。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${PROFILE:-release}"
echo "==> 构建 aibridge-ffi ($PROFILE)"
if [ "$PROFILE" = "release" ]; then
  cargo build -p aibridge-ffi --release
else
  cargo build -p aibridge-ffi
fi

# 按平台给出产物文件名
case "$(uname -s)" in
  Darwin) LIB="libaibridge.dylib" ;;
  Linux)  LIB="libaibridge.so" ;;
  MINGW*|MSYS*|CYGWIN*) LIB="aibridge.dll" ;;
  *) LIB="libaibridge.(unknown)" ;;
esac

echo "==> 完成: target/$PROFILE/$LIB"
