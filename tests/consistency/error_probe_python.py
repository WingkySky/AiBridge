"""跨语言错误一致性探针：未知 provider 必须返回 code == "provider_not_found"。

构造方式：Client(provider="nonexistent", api_key="dummy-key")。
- core 侧：api_key 校验通过 → 工厂 create_adapter 返 ProviderNotFound。
- 各语言绑定把 core AibridgeError::code() 透出为错误对象的 code 属性/字段。

期望：四种语言的错误 code 字符串完全一致，均等于 "provider_not_found"。

退出码 0 表示通过，1 表示失败。仅打印 code 与结论，不打印栈。
"""

import aibridge
from aibridge import Client, ProviderNotFoundError


def main() -> int:
    expected = "provider_not_found"
    try:
        # 未知 provider + 假 key（跳过 key 校验，触发 ProviderNotFound）
        Client(provider="nonexistent", api_key="dummy-key")
    except ProviderNotFoundError as e:
        # 异常类型正确，进一步校验消息里的 code 前缀
        msg = str(e)
        # core 错误消息格式：[provider_not_found] Provider 不存在: nonexistent
        if expected not in msg:
            print(f"[python] FAIL：消息未含 code {expected!r}，实际 {msg!r}")
            return 1
        print(f"[python] OK：异常类型=ProviderNotFoundError，消息含 code={expected!r}")
        print(f"[python]   message={msg!r}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[python] FAIL：期望 ProviderNotFoundError，实际 {type(e).__name__}: {e}")
        return 1
    print("[python] FAIL：未抛出任何异常")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
