#!/usr/bin/env bash
# ============================================================================
# AIBridge .NET 绑定构建与运行脚本
#
# 用途：
#   1. cargo build -p aibridge-ffi 产出 libaibridge 动态库
#   2. dotnet build 构建 C# 绑定（含 hello world）
#   3. dotnet run 跑通 hello world（echo 适配器，免认证）
#
# 前置：需安装 .NET 8 SDK。未装时本脚本会给出安装提示。
#   macOS:   brew install --cask dotnet-sdk
#   Linux:   见 https://learn.microsoft.com/dotnet/core/install/linux
#   Windows: 见 https://dotnet.microsoft.com/download
#
# 用法：
#   ./build.sh        # 构建 ffi + dotnet 项目
#   ./build.sh run    # 构建并运行 hello world
# ============================================================================

set -euo pipefail

# 仓库根目录（脚本位于 bindings/dotnet/build.sh，回溯三级）
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FFI_LIB_NAME="aibridge"  # P/Invoke 库名（OS 加 lib 前缀/扩展名）
DOTNET_DIR="${REPO_ROOT}/bindings/dotnet/AIBridge"

# 颜色输出
info()  { printf "\033[1;34m[INFO]\033[0m  %s\n" "$*"; }
ok()    { printf "\033[1;32m[OK]\033[0m    %s\n" "$*"; }
warn()  { printf "\033[1;33m[WARN]\033[0m  %s\n" "$*"; }
fail()  { printf "\033[1;31m[FAIL]\033[0m  %s\n" "$*"; exit 1; }

# ---------- Step 1: 构建 aibridge-ffi ----------
build_ffi() {
  info "构建 aibridge-ffi（cargo build -p aibridge-ffi）..."
  (cd "${REPO_ROOT}" && cargo build -p aibridge-ffi)

  # 确认动态库产物
  local lib_file
  if [[ "$(uname)" == "Darwin" ]]; then
    lib_file="${REPO_ROOT}/target/debug/libaibridge.dylib"
  elif [[ "$(uname)" == *MINGW* ]] || [[ "$(uname)" == *MSYS* ]]; then
    lib_file="${REPO_ROOT}/target/debug/aibridge.dll"
  else
    lib_file="${REPO_ROOT}/target/debug/libaibridge.so"
  fi

  [[ -f "${lib_file}" ]] || fail "未找到动态库产物: ${lib_file}"
  ok "FFI 动态库就绪: ${lib_file}"
}

# ---------- Step 2: 检查 dotnet ----------
check_dotnet() {
  if ! command -v dotnet &>/dev/null; then
    warn "未检测到 dotnet SDK。请先安装 .NET 8 SDK："
    echo "  macOS:   brew install --cask dotnet-sdk"
    echo "  Linux:   https://learn.microsoft.com/dotnet/core/install/linux"
    echo "  Windows: https://dotnet.microsoft.com/download"
    echo ""
    echo "代码已就绪（${DOTNET_DIR}/），装好 dotnet 后重跑本脚本即可。"
    echo "当前状态：dotnet hello world 待 dotnet 环境验证。"
    return 1
  fi
  ok "dotnet 可用: $(dotnet --version)"
}

# ---------- Step 3: dotnet build ----------
build_dotnet() {
  info "构建 .NET 项目（dotnet build）..."
  (cd "${DOTNET_DIR}" && dotnet build)
  ok ".NET 项目构建完成"
}

# ---------- Step 4: dotnet run ----------
run_hello() {
  info "运行 hello world（dotnet run）..."
  echo "----------------------------------------"
  (cd "${DOTNET_DIR}" && dotnet run --no-build)
  local rc=$?
  echo "----------------------------------------"
  if [[ ${rc} -eq 0 ]]; then
    ok "hello world 运行成功（exit 0）"
  else
    fail "hello world 运行失败（exit ${rc}）"
  fi
}

# ---------- 主流程 ----------
main() {
  info "AIBridge .NET 绑定构建脚本"
  info "仓库根: ${REPO_ROOT}"
  echo ""

  build_ffi
  echo ""

  if ! check_dotnet; then
    exit 1
  fi
  echo ""

  build_dotnet
  echo ""

  if [[ "${1:-}" == "run" ]]; then
    run_hello
  else
    info "构建完成。运行 hello world 请执行: $0 run"
  fi
}

main "$@"
