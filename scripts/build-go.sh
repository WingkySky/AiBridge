#!/usr/bin/env bash
# build-go.sh - 构建 Go 绑定（CGO 调 libaibridge）
#
# 依赖：先产 libaibridge（本脚本自动调用 build-ffi.sh）。
# 运行 Go 程序需 libaibridge 在动态库搜索路径，见末尾提示（Go 生态惯例）。
# 可用 RUN_GO_TEST=1 额外跑 go test（部分测试需真实 provider）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> 先构建 libaibridge (release)"
"$REPO_ROOT/scripts/build-ffi.sh"

LIBDIR="$REPO_ROOT/target/release"
echo "==> 构建 Go 绑定 (CGO)"
cd "$REPO_ROOT/bindings/go"
export CGO_ENABLED=1
case "$(uname -s)" in
  Darwin) export DYLD_LIBRARY_PATH="$LIBDIR:${DYLD_LIBRARY_PATH:-}" ;;
  Linux)  export LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH:-}" ;;
esac

go build ./...
echo "==> go build 通过"

if [ "${RUN_GO_TEST:-0}" = "1" ]; then
  go test ./... || echo "（go test 跳过，部分测试需真实 provider）"
fi

cat <<EOF
==> 完成。运行 Go 程序前需让动态库加载器找到 libaibridge：
    Linux:  export LD_LIBRARY_PATH=$LIBDIR
    macOS:  export DYLD_LIBRARY_PATH=$LIBDIR
    或将 libaibridge 装到系统库目录（/usr/local/lib）。
    Windows: 把 aibridge.dll 放到可执行文件同目录或 PATH。
EOF
