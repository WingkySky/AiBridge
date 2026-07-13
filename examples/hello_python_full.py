"""AIBridge Python 绑定全能力验证

用 echo mock 适配器（免认证、无网络）端到端验证全部能力：
chat / chat_stream / speech（已有）+ image / video / transcribe / embed
/ list_models / list_voices / recommend_voices（本次补全）。

每个能力打印关键字段并断言 echo 固定响应，确保 PyO3 → core 数据流正确。
"""

import asyncio

import aibridge
from aibridge import Client


async def main() -> None:
    print(f"aibridge 版本: {aibridge.__version__}")
    print("=" * 60)

    client = Client(provider="echo")
    await client.start()
    print(f"已创建客户端，provider_type={client.provider_type}")

    # --- chat（已有，回归验证）---
    print("-" * 60)
    resp = await client.chat(
        model="echo-chat",
        messages=[{"role": "user", "content": "hello"}],
    )
    content = resp.choices[0].message.content
    print(f"[chat] content = {content!r}")
    assert content == "hello [echo]", f"期望 'hello [echo]'，实际 {content!r}"

    # --- image_generate ---
    print("-" * 60)
    img = await client.image_generate(model="echo-image", prompt="cat")
    print(f"[image] id={img.id!r} model={img.model!r} data_len={len(img.data)}")
    print(f"[image] b64_json 非空: {img.data[0].b64_json is not None}")
    print(f"[image] revised_prompt={img.data[0].revised_prompt!r}")
    assert len(img.data) == 1, f"期望 1 张图，实际 {len(img.data)}"
    assert img.data[0].b64_json is not None, "echo 应返回 b64_json"
    assert img.data[0].revised_prompt == "cat", "revised_prompt 应回显 prompt"

    # --- video_create + video_poll ---
    print("-" * 60)
    task = await client.video_create(model="echo-video", prompt="cat walking")
    print(f"[video] task_id={task.task_id!r} status={task.status!r}")
    assert task.task_id == "echo-task-1", f"期望 'echo-task-1'，实际 {task.task_id!r}"
    assert task.status == "success", f"期望 'success'，实际 {task.status!r}"

    status = await client.video_poll(task.task_id)
    print(
        f"[video] poll: status={status.status!r} "
        f"video_url={status.video_url!r} progress={status.progress}"
    )
    assert status.status == "success"
    assert status.video_url == "https://example.com/echo.mp4"
    assert status.progress == 100

    # --- transcribe ---
    print("-" * 60)
    tr = await client.transcribe(model="echo-asr", file="audio.mp3")
    print(
        f"[asr] text={tr.text!r} language={tr.language!r} "
        f"duration={tr.duration} task={tr.task!r}"
    )
    assert tr.text == "echo transcription", f"期望 'echo transcription'，实际 {tr.text!r}"
    assert tr.language == "zh"
    assert tr.task == "transcribe"

    # --- embed ---
    print("-" * 60)
    emb = await client.embed(model="echo-embed", input=["hello"])
    print(
        f"[embed] model={emb.model!r} count={len(emb.data)} "
        f"prompt_tokens={emb.prompt_tokens} total_tokens={emb.total_tokens}"
    )
    vectors = emb.get_embeddings()
    print(f"[embed] get_embeddings() = {vectors}")
    assert len(emb.data) == 1, f"期望 1 个向量，实际 {len(emb.data)}"
    assert emb.data[0].index == 0
    assert len(vectors) == 1 and len(vectors[0]) == 3, "echo 应返 1 个 3 维向量"
    assert emb.prompt_tokens == 1 and emb.total_tokens == 1

    # embed 单条字符串输入
    emb_single = await client.embed(model="echo-embed", input="hello")
    assert len(emb_single.data) == 1, "单条字符串输入应返 1 个向量"

    # --- list_models ---
    print("-" * 60)
    models = await client.list_models()
    model_ids = [m.id for m in models]
    print(f"[models] 共 {len(models)} 个: {model_ids}")
    assert len(models) == 6, f"期望 6 个 echo 模型，实际 {len(models)}"
    assert "echo-chat" in model_ids and "echo-image" in model_ids
    # 验证 ModelInfo 字段
    chat_model = next(m for m in models if m.id == "echo-chat")
    print(
        f"[models] echo-chat: type={chat_model.type!r} provider={chat_model.provider!r} "
        f"supports_streaming={chat_model.supports_streaming} capabilities={chat_model.capabilities}"
    )
    assert chat_model.type == "chat"
    assert chat_model.provider == "echo"
    assert chat_model.supports_streaming is True

    # 按类型过滤
    images = await client.list_models(model_type="image")
    print(f"[models] image 过滤: {[m.id for m in images]}")
    assert len(images) == 1 and images[0].id == "echo-image"

    # --- list_voices ---
    print("-" * 60)
    voices = await client.list_voices()
    print(
        f"[voices] 共 {len(voices)} 个: "
        f"{[(v.short_name, v.locale, v.gender) for v in voices]}"
    )
    assert len(voices) == 2, f"期望 2 个音色，实际 {len(voices)}"
    assert voices[0].short_name == "echo-voice-1"
    assert voices[0].locale == "zh-CN"
    assert voices[0].gender == "Female"

    # --- recommend_voices ---
    print("-" * 60)
    # echo 的 list_voices 忽略 language 参数（返全部），默认 recommend_voices 仅按 gender 过滤；
    # 故此处验证调用成功 + limit 转发 + gender 过滤（证明绑定正确转发参数）
    all_voices = await client.recommend_voices()
    print(f"[recommend] 全部: {[(v.short_name, v.locale) for v in all_voices]}")
    assert len(all_voices) == 2, f"期望 2 个，实际 {len(all_voices)}"

    limited = await client.recommend_voices(limit=1)
    print(f"[recommend] limit=1: {[v.short_name for v in limited]}")
    assert len(limited) == 1, "limit=1 应只返 1 个"

    female = await client.recommend_voices(gender="Female")
    print(f"[recommend] gender=Female: {[(v.short_name, v.gender) for v in female]}")
    assert len(female) == 1 and female[0].gender == "Female"

    # --- speech（已有，回归验证）---
    print("-" * 60)
    result = await client.speech(model="echo-tts", input="hello", voice="alloy")
    audio = result.audio_data
    print(f"[speech] len(audio_data)={len(audio)} format={result.format!r}")
    assert len(audio) == 15, f"期望 15 字节，实际 {len(audio)}"

    # --- 关闭 ---
    print("-" * 60)
    await client.close()
    print("[close] 客户端已关闭")

    print("=" * 60)
    print("全部通过：chat / image / video / transcribe / embed / "
          "list_models / list_voices / recommend_voices / speech")


if __name__ == "__main__":
    asyncio.run(main())
