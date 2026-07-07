'use strict';

// 跨语言错误一致性探针（Node）：未知 provider 必须返回 err.code === "provider_not_found"。
//
// lib.js 的 withCode 会从 Rust map_error 编码的 "[code] message" 中解析出 .code 属性。
// 退出码 0 表示通过，1 表示失败。

const { Client } = require('../../crates/aibridge-node');

const EXPECTED = 'provider_not_found';

function main() {
  try {
    // 未知 provider + 假 key（跳过 key 校验，触发 ProviderNotFound）
    // ClientOptions 字段为 snake_case（core serde 默认，无 rename_all）
    // eslint-disable-next-line no-new
    new Client('nonexistent', { api_key: 'dummy-key' });
  } catch (err) {
    if (err.code === EXPECTED) {
      console.log(`[node] OK：err.code=${JSON.stringify(EXPECTED)}`);
      console.log(`[node]   message=${JSON.stringify(err.message)}`);
      return 0;
    }
    console.log(`[node] FAIL：期望 code=${JSON.stringify(EXPECTED)}，实际 code=${JSON.stringify(err.code)}`);
    console.log(`[node]   message=${JSON.stringify(err.message)}`);
    return 1;
  }
  console.log('[node] FAIL：未抛出任何异常');
  return 1;
}

process.exit(main());
