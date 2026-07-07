"""
AIBridge 真实 API 测试脚本

从 .env 读 API key，用 list_models() 实时拉取真实 model 名字（model 经常变，不硬编码），
测试四 provider（agnes/openai/gemini/火山）的各能力。

⚠️ 本脚本不含 API key（从 .env 读），可上传 GitHub。
   .env 含 key，已在 .gitignore，不上传 GitHub。

用法：
1. cp .env.example .env，在 .env 填入真实 API key
2. maturin develop -m crates/aibridge-python/Cargo.toml（装 aibridge 到 Python）
3. python examples/test_real_providers.py
"""

import asyncio
import os
import sys
from aibridge import Client


def load_env(path: str = ".env") -> dict:
    """读取 .env 文件（不依赖 python-dotenv，手动 parse）"""
    env = {}
    if not os.path.exists(path):
        return env
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


# 四 provider 配置：provider 名 / key 环境变量 / base_url 环境变量 / 能力
PROVIDERS = [
    {
        "name": "agnes",
        "key": "AGNES_API_KEY",
        "base_url": "AGNES_BASE_URL",
        "caps": ["chat", "image", "video", "embed"],
    },
    {
        "name": "openai",
        "key": "OPENAI_API_KEY",
        "base_url": "OPENAI_BASE_URL",
        "caps": ["chat", "image", "embed"],
    },
    {
        "name": "gemini",
        "key": "GEMINI_API_KEY",
        "base_url": "GEMINI_BASE_URL",
        "caps": ["chat", "image", "embed"],
    },
    {
        "name": "volcengine_cv",
        "key": "VOLCENGINE_CV_API_KEY",
        "base_url": "VOLCENGINE_CV_BASE_URL",
        "caps": ["image", "video"],
    },
]


def pick_model(models, model_type: str):
    """从 list_models 结果里按类型挑第一个 model（type 大小写容错）"""
    for m in models:
        try:
            if str(m.type).lower() == model_type:
                return m
        except Exception:
            continue
    return None


async def test_provider(cfg: dict, env: dict) -> None:
    name = cfg["name"]
    api_key = env.get(cfg["key"], "")
    if not api_key:
        print(f"\n=== {name} === 跳过（未填 {cfg['key']}）")
        return
    base_url = env.get(cfg["base_url"], "")
    kwargs = {"provider": name, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    print(f"\n=== {name} === (base_url={base_url or '默认'})")
    try:
        client = Client(**kwargs)
        await client.start()
    except Exception as e:
        print(f"  创建/启动失败: {e}")
        return

    try:
        # 1. list_models 实时拉取真实 model 名字
        try:
            models = await client.list_models()
            print(f"  list_models: {len(models)} 个模型")
            for m in models[:5]:
                print(f"    - {m.id} ({m.type})")
            if len(models) > 5:
                print(f"    ... 共 {len(models)} 个")
        except Exception as e:
            print(f"  list_models 失败: {e}")
            models = []

        chat_m = pick_model(models, "chat")
        image_m = pick_model(models, "image")
        video_m = pick_model(models, "video")
        embed_m = pick_model(models, "embed")

        # 2. chat
        if "chat" in cfg["caps"] and chat_m:
            try:
                r = await client.chat(
                    model=chat_m.id,
                    messages=[{"role": "user", "content": "说一个字"}],
                )
                content = r.choices[0].message.content if r.choices else "(空)"
                print(f"  chat({chat_m.id}): {str(content)[:30]}")
            except Exception as e:
                print(f"  chat({chat_m.id}) 失败: {e}")

        # 3. image
        if "image" in cfg["caps"] and image_m:
            try:
                img = await client.image_generate(
                    model=image_m.id, prompt="a cute cat"
                )
                ok = bool(img.data)
                print(f"  image({image_m.id}): {'OK' if ok else 'FAIL'}")
            except Exception as e:
                print(f"  image({image_m.id}) 失败: {e}")

        # 4. video（创建任务，不轮询，避免长时间等待）
        if "video" in cfg["caps"] and video_m:
            try:
                task = await client.video_create(
                    model=video_m.id, prompt="a cat walking"
                )
                print(f"  video({video_m.id}): task_id={task.task_id}")
            except Exception as e:
                print(f"  video({video_m.id}) 失败: {e}")

        # 5. embed
        if "embed" in cfg["caps"] and embed_m:
            try:
                e = await client.embed(model=embed_m.id, input=["hello"])
                vecs = e.get_embeddings()
                dim = len(vecs[0]) if vecs else 0
                print(f"  embed({embed_m.id}): {len(vecs)} 条, {dim} 维")
            except Exception as e:
                print(f"  embed({embed_m.id}) 失败: {e}")
    finally:
        await client.close()


async def main() -> None:
    env = load_env()
    if not env:
        print("未找到 .env，请先 cp .env.example .env 并填入 API key")
        sys.exit(1)
    print("AIBridge 真实 API 测试")
    print("（model 名由 list_models() 实时拉取，不硬编码）")
    for cfg in PROVIDERS:
        await test_provider(cfg, env)
    print("\n测试完成。如有失败项，把错误信息发给我修复。")


if __name__ == "__main__":
    asyncio.run(main())
