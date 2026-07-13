#!/usr/bin/env bash
# release.sh - 一键构建全部五语言绑定（本地用，不实际发布）
#
# 顺序：ffi（基础）→ python → node → go → jvm → dotnet
# 单语言失败不中断其它语言，最后汇总各语言结果。
# ffi 失败则整体失败（其余语言依赖 libaibridge）。
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$REPO_ROOT/scripts"

# 构建某语言并记录结果（不因失败退出）
# 用法：build_one "名称" 脚本路径
build_one() {
  local name="$1"
  local script="$2"
  echo
  echo "===================================================================="
  echo "==> 构建 $name"
  echo "===================================================================="
  if bash "$script"; then
    RESULTS+=("$name:OK")
  else
    RESULTS+=("$name:FAIL")
  fi
}

RESULTS=()
build_one "ffi (libaibridge)" "$SCRIPTS/build-ffi.sh"
build_one "Python wheel"      "$SCRIPTS/build-python.sh"
build_one "Node .node"        "$SCRIPTS/build-node.sh"
build_one "Go 绑定"           "$SCRIPTS/build-go.sh"
build_one "JVM jar"           "$SCRIPTS/build-jvm.sh"
build_one ".NET NuGet"        "$SCRIPTS/build-dotnet.sh"

echo
echo "===================================================================="
echo "==> 构建汇总"
echo "===================================================================="
for r in "${RESULTS[@]}"; do
  echo "  - $r"
done

# ffi 失败则整体失败
for r in "${RESULTS[@]}"; do
  case "$r" in
    "ffi (libaibridge):FAIL")
      echo "错误：ffi 构建失败（其它语言依赖它），整体失败" >&2
      exit 1
      ;;
  esac
done

echo "==> 完成（详见上方各语言结果）"
