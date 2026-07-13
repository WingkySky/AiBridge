#!/usr/bin/env bash
# build-jvm.sh - 构建 JVM 绑定（gradle build，含动态库打进 jar）
#
# 产物：bindings/jvm/build/libs/aibridge-jvm-*.jar
# 动态库按平台 classifier 打进 jar，放在 JNA classpath 约定路径 {os}/{arch}/
# （build.gradle.kts 的 copyNativeLib task 负责，-PembedNative=true 触发）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> 先构建 libaibridge (release)"
"$REPO_ROOT/scripts/build-ffi.sh"

echo "==> gradle build（含 native 进 jar）"
cd "$REPO_ROOT/bindings/jvm"
if [ -x ./gradlew ]; then
  ./gradlew build -PembedNative=true --no-daemon
else
  # 兜底：本地无 gradle wrapper 时用系统 gradle
  gradle build -PembedNative=true
fi

echo "==> jar 产物:"
ls -lh build/libs/*.jar 2>/dev/null || true
