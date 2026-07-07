#!/usr/bin/env bash
# build-python.sh - 构建 Python wheel（maturin）
#
# 产物：crates/aibridge-python/target/wheels/aibridge-*.whl
# macOS 默认产 universal2 wheel（同时含 amd64+arm64），需 rust target。
# 可用 BUILD_UNIVERSAL2=0 关闭 universal2。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 自动选择 Python 3.10+（pyo3 abi3-py310 要求，系统默认 python3 可能是 3.9）
if [ -z "${PYO3_PYTHON:-}" ]; then
  for py in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$py" >/dev/null 2>&1; then
      ver=$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "0.0")
      major=${ver%%.*}
      minor=${ver#*.}
      if [ "${major:-0}" -gt 3 ] || { [ "${major:-0}" -eq 3 ] && [ "${minor:-0}" -ge 10 ]; }; then
        export PYO3_PYTHON="$(command -v "$py")"
        break
      fi
    fi
  done
fi
if [ -z "${PYO3_PYTHON:-}" ]; then
  echo "错误：未找到 Python 3.10+，请安装或设置 PYO3_PYTHON 指向 3.10+ 解释器" >&2
  exit 1
fi
echo "==> 使用 Python: $PYO3_PYTHON"

# 确保 maturin 可用
if ! command -v maturin >/dev/null 2>&1; then
  echo "==> 安装 maturin"
  "$PYO3_PYTHON" -m pip install -q maturin
fi

BUILD_ARGS=(build --release)

# macOS 产 universal2 wheel（amd64+arm64 合一）
if [ "$(uname -s)" = "Darwin" ] && [ "${BUILD_UNIVERSAL2:-1}" = "1" ]; then
  echo "==> 添加 universal2 交叉编译 target（x86_64 + aarch64 apple-darwin）"
  rustup target add x86_64-apple-darwin aarch64-apple-darwin 2>/dev/null || true
  BUILD_ARGS+=(--universal2)
fi

cd "$REPO_ROOT/crates/aibridge-python"
maturin "${BUILD_ARGS[@]}"

echo "==> wheel 产物:"
ls -lh target/wheels/*.whl 2>/dev/null || true
