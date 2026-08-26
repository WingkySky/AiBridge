"""Agnes Video 2.5 / 2.5-Flash 双契约端到端测试

验证 Python 绑定 video_create 的统一参数接口：
1. 签名暴露 7 个新统一参数（duration/resolution/aspect_ratio/reference_videos/
   reference_audios/first_frame/last_frame）
2. 本地 mock 服务器实测：
   - 2.5 模型的双契约翻译（duration->seconds, resolution->size, 无 width/height 残留）
   - 轮询走 /agnesapi?video_id=&model_name= 通道，metadata.url 解析
   - Flash 参数校验（拒绝视频参考）
   - 旧模型仍走 v20 契约（ti2vid + width/height）

依赖 aiohttp 起本地 mock（与 core 的 mockito 测试互补，覆盖 Python 绑定层）。
"""

import asyncio
import inspect

import pytest
from aiohttp import web

from aibridge import Client

# 本地 mock 端口（避开常用端口段）
MOCK_PORT = 18923
MOCK_BASE_URL = f"http://127.0.0.1:{MOCK_PORT}"


def test_video_create_signature_exposes_unified_params():
    """video_create 签名应暴露全部统一参数（新旧契约共用）"""
    client = Client(provider="echo")
    sig = inspect.signature(client.video_create)
    params = list(sig.parameters.keys())
    expected = [
        "model",
        "prompt",
        "width",
        "height",
        "num_frames",
        "frame_rate",
        "mode",
        "duration",
        "resolution",
        "aspect_ratio",
        "reference_images",
        "reference_videos",
        "reference_audios",
        "first_frame",
        "last_frame",
        "negative_prompt",
        "seed",
    ]
    for p in expected:
        assert p in params, f"缺少统一参数: {p}"
    # kwargs 透传兜底仍在
    assert "kwargs" in params


@pytest.mark.asyncio
async def test_agnes_v25_dual_contract_end_to_end():
    """2.5 模型经 Python 绑定走新契约：seconds/size/aspect_ratio，无 width/height"""
    received: dict = {}

    async def create_handler(request):
        received["body"] = await request.json()
        return web.json_response({"id": "v25-mock-1", "status": "pending"})

    async def poll_handler(request):
        # 2.5 轮询必须带 video_id + model_name（agnesapi 通道）
        assert request.query.get("video_id") == "v25-mock-1"
        assert request.query.get("model_name") == "agnes-video-2.5"
        return web.json_response(
            {"status": "success", "metadata": {"url": "https://example.com/final.mp4"}}
        )

    app = web.Application()
    app.router.add_post("/videos", create_handler)
    app.router.add_get("/agnesapi", poll_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", MOCK_PORT)
    await site.start()

    try:
        client = Client(
            provider="agnes", api_key="test-key", base_url=MOCK_BASE_URL
        )
        await client.start()

        # 统一参数调用 2.5
        task = await client.video_create(
            "agnes-video-2.5",
            "一只猫在跳舞",
            duration=8,
            resolution="960P",
            aspect_ratio="16:9",
            mode="text2video",
            seed=42,
        )
        assert task.task_id == "v25-mock-1"

        b = received["body"]
        assert b["mode"] == "text", f"mode 应翻译为 text，实际 {b['mode']}"
        assert b["seconds"] == "8", f"duration 应翻译为 seconds='8'，实际 {b['seconds']}"
        assert b["size"] == "960P", f"resolution 应翻译为 size='960P'，实际 {b['size']}"
        assert b["aspect_ratio"] == "16:9"
        assert b["seed"] == 42
        assert "width" not in b, "v25 契约不应下发 width"
        assert "height" not in b, "v25 契约不应下发 height"

        # 轮询走 agnesapi 通道并解析 metadata.url
        status = await client.video_poll(task.task_id, "agnes-video-2.5")
        assert status.status == "success"
        assert status.video_url == "https://example.com/final.mp4"

        await client.close()
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_agnes_v25_flash_validation_rejects_video_reference():
    """Flash 模型传视频参考应在客户端校验阶段被拒（Validation 错误，不发请求）"""
    received: dict = {}

    async def create_handler(request):
        received["body"] = await request.json()
        return web.json_response({"id": "should-not-reach", "status": "pending"})

    app = web.Application()
    app.router.add_post("/videos", create_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", MOCK_PORT + 1)
    await site.start()

    try:
        client = Client(
            provider="agnes", api_key="test-key", base_url=f"http://127.0.0.1:{MOCK_PORT + 1}"
        )
        await client.start()

        with pytest.raises(Exception) as exc_info:
            await client.video_create(
                "agnes-video-2.5-flash",
                "test",
                mode="video2video",
                reference_videos=["https://x.com/v.mp4"],
            )
        # 校验错误应含中文提示（Validation 变体）
        assert "不支持视频参考" in str(exc_info.value)
        # 请求不应发出（校验短路）
        assert "body" not in received, "校验失败不应发出 HTTP 请求"

        await client.close()
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_agnes_v20_legacy_contract_still_works():
    """旧模型（V2.0）仍走 v20 契约：ti2vid + width/height，无 seconds/size"""
    received: dict = {}

    async def create_handler(request):
        received["body"] = await request.json()
        return web.json_response({"id": "v20-mock-1", "status": "pending"})

    app = web.Application()
    app.router.add_post("/videos", create_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", MOCK_PORT + 2)
    await site.start()

    try:
        client = Client(
            provider="agnes",
            api_key="test-key",
            base_url=f"http://127.0.0.1:{MOCK_PORT + 2}",
        )
        await client.start()

        task = await client.video_create(
            "agnes-video-v2.0", "old model", width=1920, height=1080
        )
        assert task.task_id == "v20-mock-1"

        b = received["body"]
        assert b["mode"] == "ti2vid", f"旧契约 mode 应为 ti2vid，实际 {b['mode']}"
        assert b["width"] == 1920
        assert b["height"] == 1080
        assert "seconds" not in b, "旧契约不应下发 seconds"
        assert "size" not in b, "旧契约不应下发 size"

        await client.close()
    finally:
        await runner.cleanup()
