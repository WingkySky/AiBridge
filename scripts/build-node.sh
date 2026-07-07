#!/usr/bin/env bash
# build-node.sh - 构建 Node.js 原生模块（napi-rs .node）
#
# 产物：crates/aibridge-node/aibridge.{platform}-{arch}.node
# 多平台发布：每个平台在 CI 矩阵中分别构建，由 `napi prepublish` 汇总为
# optionalDependencies 子包（见 package.json 的 napi.triples 配置）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/crates/aibridge-node"

echo "==> npm install"
[ -d node_modules ] || npm install

echo "==> napi build (release, --platform)"
npx napi build --platform --release

echo "==> .node 产物:"
ls -lh *.node 2>/dev/null || true
