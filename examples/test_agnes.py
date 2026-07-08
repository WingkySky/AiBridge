"""临时：只测 agnes。测完可删。自动选模型。"""
import asyncio
from aibridge import Client


def load_env():
    env = {}
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


async def main():
    env = load_env()
    c = Client(
        provider="agnes",
        api_key=env["AGNES_API_KEY"],
        base_url=env["AGNES_BASE_URL"],
    )
    await c.start()

    # 1. list_models
    print("=== list_models ===")
    models = []
    try:
        models = await c.list_models()
        print(f"✅ 通：{len(models)} 个模型")
        for m in models:
            print(f"  - {m.id} ({m.type})")
    except Exception as e:
        print(f"❌ 失败: {e}")

    # 2. chat：自动选第一个 chat model
    chat_m = next((m for m in models if str(m.type).lower() == "chat"), None)
    if chat_m:
        print(f"\n=== chat({chat_m.id}) ===")
        try:
            r = await c.chat(
                model=chat_m.id, messages=[{"role": "user", "content": "说一个字"}]
            )
            content = r.choices[0].message.content if r.choices else "(空)"
            print(f"✅ 通: {str(content)[:30]}")
        except Exception as e:
            print(f"❌ 失败: {e}")

    # 3. image
    image_m = next((m for m in models if str(m.type).lower() == "image"), None)
    if image_m:
        print(f"\n=== image({image_m.id}) ===")
        try:
            img = await c.image_generate(model=image_m.id, prompt="一只猫")
            print(f"✅ 通: {'有数据' if img.data else '无数据'}")
        except Exception as e:
            print(f"❌ 失败: {e}")

    # 4. video
    video_m = next((m for m in models if str(m.type).lower() == "video"), None)
    if video_m:
        print(f"\n=== video({video_m.id}) ===")
        try:
            task = await c.video_create(model=video_m.id, prompt="一只猫在走路")
            print(f"✅ 通: task_id={task.task_id}")
        except Exception as e:
            print(f"❌ 失败: {e}")

    await c.close()


if __name__ == "__main__":
    asyncio.run(main())
