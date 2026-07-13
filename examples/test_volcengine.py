"""临时：只测火山引擎（国内可直连）。测完可删。

完全自动：list_models 后选第一个 image/video 模型测试，不硬编码 model 名。
只报告每项通/不通。
"""
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
        provider="volcengine_cv",
        api_key=env["VOLCENGINE_CV_API_KEY"],
        base_url=env["VOLCENGINE_CV_BASE_URL"],
    )
    await c.start()

    # 1. list_models（验证连通性 + 认证 + 代码）
    print("=== list_models ===")
    models = []
    try:
        models = await c.list_models()
        print(f"✅ 通：{len(models)} 个模型")
        for t in ["chat", "image", "video", "embed"]:
            m = next((m for m in models if str(m.type).lower() == t), None)
            if m:
                print(f"  {t}: {m.id}")
    except Exception as e:
        print(f"❌ 失败: {e}")
        await c.close()
        return

    # 2. image：自动选第一个 image model
    image_m = next((m for m in models if str(m.type).lower() == "image"), None)
    if image_m:
        print(f"\n=== image_generate({image_m.id}) ===")
        try:
            img = await c.image_generate(model=image_m.id, prompt="一只猫")
            print(f"✅ 通：{'有数据' if img.data else '无数据'}")
        except Exception as e:
            print(f"❌ 失败: {e}")
    else:
        print("\nimage：list 无 image 模型，跳过")

    # 3. video：自动选第一个 video model
    video_m = next((m for m in models if str(m.type).lower() == "video"), None)
    if video_m:
        print(f"\n=== video_create({video_m.id}) ===")
        try:
            task = await c.video_create(model=video_m.id, prompt="一只猫在走路")
            print(f"✅ 通：task_id={task.task_id}")
        except Exception as e:
            print(f"❌ 失败: {e}")
    else:
        print("\nvideo：list 无 video 模型，跳过")

    await c.close()


if __name__ == "__main__":
    asyncio.run(main())
