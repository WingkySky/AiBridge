"""
AGN-SDK 语音功能适配器单元测试

覆盖：
- 中文模型：Qwen(通义千问)、Doubao(豆包)、MiniMax(稀宇科技)
- 聚合平台：SiliconFlow(硅基流动)、Together AI、Fireworks AI
- OpenAI 官方：whisper-1、tts-1、tts-1-hd
- Groq: Whisper Large v3 / Turbo（LPU 极速 ASR）
- ElevenLabs: 全球主流 TTS 高质量音色
- Deepgram: Nova-2/Nova-3 高速 ASR
- 测试方法：transcribe()、speech()、list_models()、get_audio_bytes()
- 使用 unittest.mock 模拟 HTTP 请求，验证请求构造和响应解析
"""

import io
import base64
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agn.adapters.chinese import (
    QwenAdapter,
    DoubaoAdapter,
    MiniMaxAdapter,
)
from agn.adapters.aggregation_platforms import (
    SiliconFlowAdapter,
    TogetherAIAdapter,
    FireworksAIAdapter,
)
from agn.adapters.openai import OpenAIAdapter
from agn.adapters.additional_models import GroqAdapter
from agn.adapters.audio_adapters import (
    ElevenLabsAdapter,
    DeepgramAdapter,
    AssemblyAIAdapter,
    CartesiaAdapter,
    EdgeTTSAdapter,
)
from agn.adapters.base import Capabilities
from agn.models.common import ProviderConfig
from agn.models.audio import TranscriptionResult, SpeechResult


# OpenAI 兼容 ASR/TTS 适配器（继承 OpenAICompatibleAudioMixin）
AUDIO_ADAPTERS = [
    QwenAdapter,
    DoubaoAdapter,
    MiniMaxAdapter,
    SiliconFlowAdapter,
    TogetherAIAdapter,
    FireworksAIAdapter,
    OpenAIAdapter,
    GroqAdapter,
]

# 非 Mixin 的专用 ASR 适配器（自己实现 get_audio_bytes/transcribe）
STANDALONE_ASR_ADAPTERS = [
    DeepgramAdapter,
    AssemblyAIAdapter,
]

# 同时支持 ASR 和 TTS 的适配器
ADAPTERS_WITH_TTS = [
    QwenAdapter,
    DoubaoAdapter,
    MiniMaxAdapter,
    SiliconFlowAdapter,
    TogetherAIAdapter,
    OpenAIAdapter,
]

# 仅支持 TTS 不支持 ASR 的适配器
TTS_ONLY_ADAPTERS = [
    ElevenLabsAdapter,
    CartesiaAdapter,
    EdgeTTSAdapter,
]

# 仅支持 ASR 不支持 TTS 的适配器（OpenAI 兼容 + 独立实现）
ASR_ONLY_ADAPTERS = [
    FireworksAIAdapter,
    GroqAdapter,
    DeepgramAdapter,
    AssemblyAIAdapter,
]

# 所有支持 ASR 的适配器
ALL_ASR_ADAPTERS = AUDIO_ADAPTERS + STANDALONE_ASR_ADAPTERS


class TestAudioCapabilitiesDeclaration:
    """测试所有适配器语音能力声明"""

    @pytest.mark.parametrize("adapter_cls", AUDIO_ADAPTERS + STANDALONE_ASR_ADAPTERS)
    def test_adapter_has_audio_capabilities(self, adapter_cls, mock_api_key):
        """测试适配器声明了语音转文字能力；全功能适配器还声明文字转语音能力"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)

        assert Capabilities.AUDIO_TRANSCRIBE in adapter.supported_capabilities, \
            f"{adapter_cls.__name__} 缺少 AUDIO_TRANSCRIBE 能力"
        assert adapter.supports_capability(Capabilities.AUDIO_TRANSCRIBE), \
            f"{adapter_cls.__name__} supports_capability(audio_transcribe) 返回 False"

        if adapter_cls not in ASR_ONLY_ADAPTERS:
            assert Capabilities.AUDIO_SPEECH in adapter.supported_capabilities, \
                f"{adapter_cls.__name__} 缺少 AUDIO_SPEECH 能力"
            assert adapter.supports_capability(Capabilities.AUDIO_SPEECH), \
                f"{adapter_cls.__name__} supports_capability(audio_speech) 返回 False"
        else:
            assert Capabilities.AUDIO_SPEECH not in adapter.supported_capabilities, \
                f"{adapter_cls.__name__} 不应支持 AUDIO_SPEECH（ASR-only）"

    @pytest.mark.parametrize("adapter_cls", TTS_ONLY_ADAPTERS)
    def test_tts_only_adapter_capabilities(self, adapter_cls, mock_api_key):
        """测试仅支持 TTS 的适配器（ElevenLabs）"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)

        assert Capabilities.AUDIO_SPEECH in adapter.supported_capabilities, \
            f"{adapter_cls.__name__} 缺少 AUDIO_SPEECH 能力"
        assert adapter.supports_capability(Capabilities.AUDIO_SPEECH), \
            f"{adapter_cls.__name__} supports_capability(audio_speech) 返回 False"
        assert Capabilities.AUDIO_TRANSCRIBE not in adapter.supported_capabilities, \
            f"{adapter_cls.__name__} 不应支持 AUDIO_TRANSCRIBE（TTS-only）"

    @pytest.mark.parametrize("adapter_cls", AUDIO_ADAPTERS + STANDALONE_ASR_ADAPTERS)
    def test_adapter_has_audio_methods(self, adapter_cls, mock_api_key):
        """测试 ASR 适配器实例有 transcribe 和 get_audio_bytes 方法"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)

        assert hasattr(adapter, "transcribe"), "缺少 transcribe 方法"
        assert callable(adapter.transcribe), "transcribe 不是可调用方法"
        assert hasattr(adapter, "get_audio_bytes"), "缺少 get_audio_bytes 方法"
        assert callable(adapter.get_audio_bytes), "get_audio_bytes 不是可调用方法"

        if adapter_cls not in ASR_ONLY_ADAPTERS:
            assert hasattr(adapter, "speech"), "缺少 speech 方法"
            assert callable(adapter.speech), "speech 不是可调用方法"

    @pytest.mark.parametrize("adapter_cls", TTS_ONLY_ADAPTERS)
    def test_tts_only_adapter_methods(self, adapter_cls, mock_api_key):
        """测试 TTS-only 适配器有 speech 方法，transcribe 抛出异常"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)

        assert hasattr(adapter, "speech"), "缺少 speech 方法"
        assert callable(adapter.speech), "speech 不是可调用方法"
        assert hasattr(adapter, "transcribe"), "缺少 transcribe 方法"
        assert callable(adapter.transcribe), "transcribe 不是可调用方法"


class TestAudioModelsInList:
    """测试语音模型在 list_models 中正确列出"""

    @pytest.mark.parametrize("adapter_cls,expected_asr_models,expected_tts_models", [
        (QwenAdapter, ["sensevoice-v1"], ["cosyvoice-v1"]),
        (DoubaoAdapter, ["doubao-asr"], ["doubao-tts"]),
        (MiniMaxAdapter, ["abab-asr"], ["abab-tts", "speech-01"]),
        (SiliconFlowAdapter, ["FunAudioLLM/SenseVoiceSmall", "iic/SenseVoiceSmall"], []),
        (TogetherAIAdapter, ["openai/whisper-large-v3", "openai/whisper-large-v3-turbo"], []),
        (FireworksAIAdapter, ["accounts/fireworks/models/whisper-v3"], []),
        (OpenAIAdapter, ["whisper-1"], ["tts-1", "tts-1-hd"]),
        (GroqAdapter, ["whisper-large-v3", "whisper-large-v3-turbo", "distil-whisper-large-v3-en"], []),
    ])
    @pytest.mark.asyncio
    async def test_list_models_includes_audio_models(
        self, adapter_cls, expected_asr_models, expected_tts_models, mock_api_key
    ):
        """测试列表包含预期的语音模型"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)
        models = await adapter.list_models()

        model_ids = {m.id for m in models}
        model_capabilities = {m.id: m.capabilities for m in models}

        for model_id in expected_asr_models:
            assert model_id in model_ids, f"模型列表缺少 ASR 模型: {model_id}"
            assert "audio_transcribe" in model_capabilities[model_id], \
                f"模型 {model_id} 缺少 audio_transcribe 能力标记"

        for model_id in expected_tts_models:
            assert model_id in model_ids, f"模型列表缺少 TTS 模型: {model_id}"
            assert "audio_speech" in model_capabilities[model_id], \
                f"模型 {model_id} 缺少 audio_speech 能力标记"

    @pytest.mark.parametrize("adapter_cls,expected_tts_models", [
        (ElevenLabsAdapter, [
            "eleven_multilingual_v2",
            "eleven_multilingual_v1",
            "eleven_monolingual_v1",
            "eleven_turbo_v2_5",
            "eleven_turbo_v2",
            "eleven_flash_v2_5",
        ]),
    ])
    @pytest.mark.asyncio
    async def test_list_models_tts_only_adapters(self, adapter_cls, expected_tts_models, mock_api_key):
        """测试 TTS-only 适配器的模型列表"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)
        models = await adapter.list_models()

        model_ids = {m.id for m in models}
        model_capabilities = {m.id: m.capabilities for m in models}

        for model_id in expected_tts_models:
            assert model_id in model_ids, f"模型列表缺少 TTS 模型: {model_id}"
            assert "audio_speech" in model_capabilities[model_id], \
                f"模型 {model_id} 缺少 audio_speech 能力标记"

    @pytest.mark.parametrize("adapter_cls,expected_asr_models", [
        (DeepgramAdapter, [
            "nova-3", "nova-2", "nova-2-general", "nova-2-meeting",
            "nova-2-phonecall", "nova-2-conversationalai", "nova-2-video",
            "nova-2-medical", "nova-2-finance", "nova-2-drivethru",
            "whisper-large", "whisper-medium", "whisper-small",
            "enhanced", "base",
        ]),
    ])
    @pytest.mark.asyncio
    async def test_list_models_asr_only_standalone_adapters(self, adapter_cls, expected_asr_models, mock_api_key):
        """测试独立 ASR 适配器（Deepgram）的模型列表"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)
        models = await adapter.list_models()

        model_ids = {m.id for m in models}
        model_capabilities = {m.id: m.capabilities for m in models}

        for model_id in expected_asr_models:
            assert model_id in model_ids, f"模型列表缺少 ASR 模型: {model_id}"
            assert "audio_transcribe" in model_capabilities[model_id], \
                f"模型 {model_id} 缺少 audio_transcribe 能力标记"

    @pytest.mark.parametrize("adapter_cls", ALL_ASR_ADAPTERS)
    @pytest.mark.asyncio
    async def test_list_audio_type_models(self, adapter_cls, mock_api_key):
        """测试按 type='audio' 过滤只返回语音模型"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)
        models = await adapter.list_models(model_type="audio")

        assert len(models) > 0, "应该有语音模型"
        for m in models:
            assert m.type == "audio", f"模型 {m.id} 的 type 应该是 audio，实际是 {m.type}"

    @pytest.mark.parametrize("adapter_cls", TTS_ONLY_ADAPTERS)
    @pytest.mark.asyncio
    async def test_list_audio_type_models_tts_only(self, adapter_cls, mock_api_key):
        """测试 TTS-only 适配器按 type='audio' 过滤"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)
        models = await adapter.list_models(model_type="audio")

        assert len(models) > 0, "应该有语音模型"
        for m in models:
            assert m.type == "audio", f"模型 {m.id} 的 type 应该是 audio，实际是 {m.type}"
            assert "audio_speech" in m.capabilities


class TestGetAudioBytes:
    """测试 get_audio_bytes 方法（核心工具方法）"""

    @pytest.mark.parametrize("adapter_cls", AUDIO_ADAPTERS)
    @pytest.mark.asyncio
    async def test_get_audio_bytes_from_bytes(self, adapter_cls, mock_api_key):
        """测试从 bytes 读取音频数据"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)

        test_data = b"hello wav bytes"
        data, filename = await adapter.get_audio_bytes(test_data)

        assert data == test_data
        assert filename.endswith(".wav") or filename.endswith(".mp3")
        assert isinstance(filename, str)

    @pytest.mark.parametrize("adapter_cls", AUDIO_ADAPTERS)
    @pytest.mark.asyncio
    async def test_get_audio_bytes_from_file_like_object(self, adapter_cls, mock_api_key):
        """测试从类文件对象读取音频"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)

        test_data = b"file like content"
        file_obj = io.BytesIO(test_data)
        file_obj.name = "test_audio.mp3"

        data, filename = await adapter.get_audio_bytes(file_obj)
        assert data == test_data
        assert "test_audio.mp3" in filename or filename.endswith(".mp3")

    @pytest.mark.parametrize("adapter_cls", AUDIO_ADAPTERS)
    @pytest.mark.asyncio
    async def test_get_audio_bytes_from_file_path(self, adapter_cls, mock_api_key, tmp_path):
        """测试从文件路径读取音频"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)

        test_data = b"file path content"
        test_file = tmp_path / "recording.wav"
        test_file.write_bytes(test_data)

        data, filename = await adapter.get_audio_bytes(str(test_file))
        assert data == test_data
        assert "recording.wav" in filename

    @pytest.mark.parametrize("adapter_cls", AUDIO_ADAPTERS)
    @pytest.mark.asyncio
    async def test_get_audio_bytes_from_base64(self, adapter_cls, mock_api_key):
        """测试从 base64 字符串读取音频"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)

        original_data = b"base64 encoded audio data"
        b64_str = base64.b64encode(original_data).decode("utf-8")

        data, filename = await adapter.get_audio_bytes(b64_str)
        assert data == original_data
        assert filename.endswith(".wav") or filename.endswith(".mp3")

    @pytest.mark.parametrize("adapter_cls", AUDIO_ADAPTERS)
    @pytest.mark.asyncio
    async def test_get_audio_bytes_from_base64_with_prefix(self, adapter_cls, mock_api_key):
        """测试从带 data:audio/...;base64, 前缀的 base64 读取"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)

        original_data = b"prefixed base64"
        b64_str = base64.b64encode(original_data).decode("utf-8")
        prefixed = f"data:audio/wav;base64,{b64_str}"

        data, filename = await adapter.get_audio_bytes(prefixed)
        assert data == original_data
        assert filename.endswith(".wav")

    @pytest.mark.parametrize("adapter_cls", AUDIO_ADAPTERS)
    @pytest.mark.asyncio
    async def test_get_audio_bytes_unsupported_type(self, adapter_cls, mock_api_key):
        """测试不支持的类型抛出异常"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)

        with pytest.raises(ValueError, match="Unsupported file type"):
            await adapter.get_audio_bytes(12345)

        with pytest.raises((ValueError, FileNotFoundError, OSError)):
            await adapter.get_audio_bytes("/nonexistent/file.wav")


class TestTranscribeMethod:
    """测试语音转文字方法逻辑（Mock HTTP）"""

    def _mock_response(self, json_data, status_code=200):
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=json_data)
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    @pytest.mark.parametrize("adapter_cls", AUDIO_ADAPTERS)
    @pytest.mark.asyncio
    async def test_transcribe_with_bytes(self, adapter_cls, mock_api_key):
        """测试使用 bytes 音频输入调用转写接口"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)
        await adapter.start()

        mock_result = {
            "text": "你好，世界",
            "language": "zh",
            "duration": 2.5,
        }

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            test_audio = b"fake wav data"
            result = await adapter.transcribe(
                model="test-asr-model",
                file=test_audio,
                language="zh",
            )

            mock_post.assert_called_once()

            assert isinstance(result, TranscriptionResult)
            assert result.text == "你好，世界"
            assert result.language == "zh"
            assert result.duration == 2.5
            assert result.task == "transcribe"
            assert result.model == "test-asr-model"

        await adapter.close()

    @pytest.mark.parametrize("adapter_cls", AUDIO_ADAPTERS)
    @pytest.mark.asyncio
    async def test_transcribe_with_segments_and_words(self, adapter_cls, mock_api_key):
        """测试解析带 segments 和 words 的详细响应"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)
        await adapter.start()

        mock_result = {
            "text": "测试分段",
            "language": "zh",
            "duration": 1.0,
            "segments": [
                {"id": 0, "seek": 0, "start": 0.0, "end": 1.0, "text": "测试分段", "tokens": []}
            ],
            "words": [
                {"word": "测试", "start": 0.0, "end": 0.5},
                {"word": "分段", "start": 0.5, "end": 1.0},
            ],
        }

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.transcribe(
                model="test-asr",
                file=b"audio data",
                response_format="verbose_json",
            )

            assert result.text == "测试分段"
            assert result.segments is not None
            assert len(result.segments) == 1
            assert result.segments[0].text == "测试分段"
            assert result.words is not None
            assert len(result.words) == 2
            assert result.words[0].word == "测试"

        await adapter.close()

    @pytest.mark.parametrize("adapter_cls", AUDIO_ADAPTERS)
    @pytest.mark.asyncio
    async def test_transcribe_unsupported_file_type_raises(self, adapter_cls, mock_api_key):
        """测试不支持的文件类型抛出异常"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)
        await adapter.start()

        with pytest.raises(ValueError, match="Unsupported file type"):
            await adapter.transcribe(model="test", file=12345)

        await adapter.close()

    @pytest.mark.asyncio
    async def test_translate_method_openai(self, mock_api_key):
        """测试 OpenAI translate 方法（语音翻译）"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = OpenAIAdapter(config=config)
        await adapter.start()

        mock_result = {
            "text": "Hello, world",
            "duration": 2.0,
        }

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.translate(
                model="whisper-1",
                file=b"chinese audio",
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/audio/translations" in call_args.args[0]

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, world"
            assert result.task == "translate"
            assert result.language == "en"
            assert result.model == "whisper-1"

        await adapter.close()


class TestSpeechMethod:
    """测试文字转语音方法逻辑（Mock HTTP）"""

    def _mock_speech_response(self, audio_data=b"fake mp3 data", format="mp3"):
        """创建模拟 TTS 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = audio_data
        mock_resp.headers = {"content-type": f"audio/{format}"}
        return mock_resp

    @pytest.mark.parametrize("adapter_cls", ADAPTERS_WITH_TTS)
    @pytest.mark.asyncio
    async def test_speech_basic(self, adapter_cls, mock_api_key):
        """测试基础语音合成请求"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)
        await adapter.start()

        test_audio = b"fake tts audio"

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_speech_response(test_audio)

            result = await adapter.speech(
                model="test-tts-model",
                input="你好，欢迎使用语音合成",
                voice="alloy",
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/audio/speech" in call_args.args[0]

            json_body = call_args.kwargs.get("json", {})
            assert json_body["model"] == "test-tts-model"
            assert json_body["input"] == "你好，欢迎使用语音合成"
            assert json_body["voice"] == "alloy"

            assert isinstance(result, SpeechResult)
            assert result.audio_data == test_audio
            assert result.format == "mp3"
            assert result.model == "test-tts-model"
            assert "audio" in result.content_type

        await adapter.close()

    @pytest.mark.parametrize("adapter_cls", ADAPTERS_WITH_TTS)
    @pytest.mark.asyncio
    async def test_speech_with_options(self, adapter_cls, mock_api_key):
        """测试带可选参数（语速、格式）的语音合成"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = adapter_cls(config=config)
        await adapter.start()

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_speech_response(b"wav data", format="wav")

            result = await adapter.speech(
                model="test-tts",
                input="测试语速调整",
                voice="nova",
                response_format="wav",
                speed=1.2,
            )

            json_body = mock_post.call_args.kwargs.get("json", {})
            assert json_body["response_format"] == "wav"
            assert json_body["speed"] == 1.2
            assert result.format == "wav"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_speech_save_to_file(self, tmp_path, mock_api_key):
        """测试 SpeechResult.save_to_file 方法"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = OpenAIAdapter(config=config)
        await adapter.start()

        test_data = b"test audio content for saving"

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_speech_response(test_data)

            result = await adapter.speech(
                model="tts-1",
                input="测试保存到文件",
                voice="alloy",
            )

            save_path = tmp_path / "test_output.mp3"
            result.save_to_file(str(save_path))

            assert save_path.exists()
            saved_content = save_path.read_bytes()
            assert saved_content == test_data

        await adapter.close()

    @pytest.mark.asyncio
    async def test_openai_tts_models(self, mock_api_key):
        """测试 OpenAI TTS 特定模型"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = OpenAIAdapter(config=config)
        await adapter.start()

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_speech_response()

            result = await adapter.speech(
                model="tts-1-hd",
                input="测试高清语音",
                voice="echo",
            )

            json_body = mock_post.call_args.kwargs.get("json", {})
            assert json_body["model"] == "tts-1-hd"
            assert result.model == "tts-1-hd"

        await adapter.close()


class TestRouterAudioMapping:
    """测试 Router 中语音模型路由映射"""

    def test_router_has_audio_model_mappings(self):
        """测试 MODEL_PROVIDER_MAP 包含所有新增语音模型"""
        from agn.router import Router

        expected_mappings = {
            "whisper-1": "openai",
            "tts-1": "openai",
            "tts-1-hd": "openai",
            "sensevoice-v1": "qwen",
            "cosyvoice-v1": "qwen",
            "doubao-asr": "doubao",
            "doubao-tts": "doubao",
            "abab-asr": "minimax",
            "abab-tts": "minimax",
            "speech-01": "minimax",
            "FunAudioLLM/SenseVoiceSmall": "siliconflow",
            "iic/SenseVoiceSmall": "siliconflow",
            "openai/whisper-large-v3": "togetherai",
            "openai/whisper-large-v3-turbo": "togetherai",
            "accounts/fireworks/models/whisper-v3": "fireworksai",
        }

        MODEL_PROVIDER_MAP = Router.MODEL_PROVIDER_MAP

        for model_id, expected_provider in expected_mappings.items():
            assert model_id in MODEL_PROVIDER_MAP, \
                f"路由表缺少模型映射: {model_id}"
            assert MODEL_PROVIDER_MAP[model_id] == expected_provider, \
                f"模型 {model_id} 应该映射到 {expected_provider}，" \
                f"实际是 {MODEL_PROVIDER_MAP[model_id]}"


class TestTranscribeEdgeCases:
    """测试语音转文字边缘情况"""

    def _mock_response(self, json_data, status_code=200):
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=json_data)
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    @pytest.mark.asyncio
    async def test_transcribe_with_file_like_object(self, mock_api_key, tmp_path):
        """测试使用文件对象进行转写"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = OpenAIAdapter(config=config)
        await adapter.start()

        mock_result = {"text": "文件对象测试", "language": "zh"}
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"fake wav content")

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            with open(test_file, "rb") as f:
                result = await adapter.transcribe(
                    model="whisper-1",
                    file=f,
                )

            assert result.text == "文件对象测试"
            call_files = mock_post.call_args.kwargs.get("files", {})
            assert "file" in call_files

        await adapter.close()

    @pytest.mark.asyncio
    async def test_transcribe_with_extra_parameters(self, mock_api_key):
        """测试转写时传入额外参数（prompt, temperature 等）"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = DoubaoAdapter(config=config)
        await adapter.start()

        mock_result = {"text": "参数测试"}

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            await adapter.transcribe(
                model="doubao-asr",
                file=b"audio",
                prompt="这是专业术语",
                temperature=0.2,
                response_format="json",
            )

            call_data = mock_post.call_args.kwargs.get("data", {})
            assert call_data.get("prompt") == "这是专业术语"
            assert call_data.get("temperature") == 0.2
            assert call_data.get("response_format") == "json"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_transcribe_with_base64_input(self, mock_api_key):
        """测试使用 base64 音频输入"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = QwenAdapter(config=config)
        await adapter.start()

        mock_result = {"text": "base64 测试"}
        test_audio = b"base64 audio data"
        b64_audio = base64.b64encode(test_audio).decode("utf-8")

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.transcribe(
                model="sensevoice-v1",
                file=b64_audio,
            )

            assert result.text == "base64 测试"
            call_files = mock_post.call_args.kwargs.get("files", {})
            assert "file" in call_files

        await adapter.close()


class TestSpeechEdgeCases:
    """测试文字转语音边缘情况"""

    def _mock_speech_response(self, audio_data=b"fake data", format="mp3"):
        """创建模拟 TTS 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = audio_data
        mock_resp.headers = {"content-type": f"audio/{format}"}
        return mock_resp

    @pytest.mark.asyncio
    async def test_speech_multiple_formats(self, mock_api_key):
        """测试生成不同音频格式"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = OpenAIAdapter(config=config)
        await adapter.start()

        formats = ["opus", "aac", "flac", "wav", "pcm"]
        for fmt in formats:
            with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = self._mock_speech_response(b"data", format=fmt)
                result = await adapter.speech(
                    model="tts-1",
                    input="多格式测试",
                    voice="alloy",
                    response_format=fmt,
                )
                assert result.format == fmt

        await adapter.close()

    @pytest.mark.asyncio
    async def test_fireworks_no_speech_capability(self, mock_api_key):
        """测试 Fireworks AI 没有 speech 能力"""
        config = ProviderConfig(provider_type="test", api_key=mock_api_key)
        adapter = FireworksAIAdapter(config=config)

        assert not adapter.supports_capability(Capabilities.AUDIO_SPEECH)
        assert Capabilities.AUDIO_SPEECH not in adapter.supported_capabilities


class TestAudioDataModels:
    """测试语音数据模型"""

    def test_transcription_result_defaults(self):
        """测试 TranscriptionResult 默认值"""
        result = TranscriptionResult(text="测试文本")
        assert result.text == "测试文本"
        assert result.language is None
        assert result.duration is None
        assert result.segments is None
        assert result.words is None
        assert result.task == "transcribe"
        assert result.model is None

    def test_transcription_result_with_all_fields(self):
        """测试 TranscriptionResult 完整字段"""
        from agn.models.audio import TranscriptionSegment, TranscriptionWord
        result = TranscriptionResult(
            text="完整测试",
            language="en",
            duration=5.0,
            segments=[TranscriptionSegment(id=0, start=0.0, end=5.0, text="完整测试")],
            words=[TranscriptionWord(word="完整", start=0.0, end=2.5)],
            task="translate",
            model="whisper-1",
        )
        assert result.text == "完整测试"
        assert result.language == "en"
        assert result.duration == 5.0
        assert len(result.segments) == 1
        assert len(result.words) == 1
        assert result.task == "translate"
        assert result.model == "whisper-1"

    def test_speech_result_has_save_method(self):
        """测试 SpeechResult 有 save_to_file 方法"""
        result = SpeechResult(
            audio_data=b"audio",
            content_type="audio/mpeg",
            format="mp3",
            model="test",
        )
        assert hasattr(result, "save_to_file")
        assert callable(result.save_to_file)

    def test_speech_result_default_format(self):
        """测试 SpeechResult 默认格式"""
        result = SpeechResult(audio_data=b"test")
        assert result.format == "mp3"
        assert result.content_type == "audio/mpeg"
        assert result.model is None

    def test_segment_defaults(self):
        """测试 TranscriptionSegment 默认值"""
        from agn.models.audio import TranscriptionSegment
        seg = TranscriptionSegment(id=0, start=0.0, end=1.0, text="test")
        assert seg.id == 0
        assert seg.avg_logprob is None
        assert seg.compression_ratio is None
        assert seg.no_speech_prob is None
        assert seg.temperature is None
        assert seg.tokens is None
        assert seg.seek is None

    def test_word_structure(self):
        """测试 TranscriptionWord 结构"""
        from agn.models.audio import TranscriptionWord
        word = TranscriptionWord(word="你好", start=0.0, end=0.5)
        assert word.word == "你好"
        assert word.start == 0.0
        assert word.end == 0.5


# ==================== Groq Whisper ASR 专用测试 ====================


class TestGroqWhisperASR:
    """测试 Groq Whisper 高速语音识别"""

    def _mock_response(self, json_data, status_code=200):
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=json_data)
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    @pytest.mark.asyncio
    async def test_groq_whisper_transcribe_basic(self, mock_api_key):
        """测试 Groq Whisper Large v3 基础转写"""
        config = ProviderConfig(provider_type="groq", api_key=mock_api_key)
        adapter = GroqAdapter(config=config)
        await adapter.start()

        mock_result = {
            "text": "Groq LPU 极速语音识别测试",
            "language": "zh",
            "duration": 1.5,
        }

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.transcribe(
                model="whisper-large-v3",
                file=b"fake audio data",
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/audio/transcriptions" in call_args.args[0]

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Groq LPU 极速语音识别测试"
            assert result.language == "zh"
            assert result.model == "whisper-large-v3"
            assert result.duration == 1.5

        await adapter.close()

    @pytest.mark.asyncio
    async def test_groq_whisper_turbo_transcribe(self, mock_api_key):
        """测试 Groq Whisper Large v3 Turbo 极速转写"""
        config = ProviderConfig(provider_type="groq", api_key=mock_api_key)
        adapter = GroqAdapter(config=config)
        await adapter.start()

        mock_result = {
            "text": "Turbo 极速测试",
            "x_groq": {"id": "test-id"},
        }

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.transcribe(
                model="whisper-large-v3-turbo",
                file=b"audio",
            )

            assert result.text == "Turbo 极速测试"
            assert result.model == "whisper-large-v3-turbo"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_groq_speech_raises_unsupported(self, mock_api_key):
        """测试 Groq 不支持 TTS 时抛出异常"""
        from agn.core.errors import UnsupportedCapabilityError

        config = ProviderConfig(provider_type="groq", api_key=mock_api_key)
        adapter = GroqAdapter(config=config)

        with pytest.raises(UnsupportedCapabilityError, match="text-to-speech"):
            await adapter.speech(
                model="some-model",
                input="测试",
                voice="test",
            )

    @pytest.mark.asyncio
    async def test_groq_translate(self, mock_api_key):
        """测试 Groq Whisper 翻译（translate）"""
        config = ProviderConfig(provider_type="groq", api_key=mock_api_key)
        adapter = GroqAdapter(config=config)
        await adapter.start()

        mock_result = {
            "text": "This is a translation test",
            "language": "en",
        }

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.translate(
                model="whisper-large-v3",
                file=b"chinese audio",
            )

            call_args = mock_post.call_args
            assert "/audio/translations" in call_args.args[0]
            assert result.text == "This is a translation test"
            assert result.task == "translate"

        await adapter.close()


# ==================== ElevenLabs TTS 专用测试 ====================


class TestElevenLabsTTS:
    """测试 ElevenLabs 高质量语音合成"""

    def _mock_speech_response(self, audio_data=b"fake mp3", content_type="audio/mpeg"):
        """创建模拟 TTS 响应（二进制音频）"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = audio_data
        mock_resp.headers = {"content-type": content_type}
        return mock_resp

    def _mock_error_response(self, status_code, json_data=None):
        """创建模拟错误响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=json_data or {})
        mock_resp.text = str(json_data) if json_data else f"HTTP {status_code}"
        return mock_resp

    def test_elevenlabs_default_voices_mapping(self):
        """测试 ElevenLabs 内置音色映射"""
        config = ProviderConfig(provider_type="elevenlabs", api_key="test-key")
        adapter = ElevenLabsAdapter(config=config)

        rachel_id = adapter._get_voice_id("Rachel")
        assert rachel_id == "21m00Tcm4TlvDq8ikWAM"

        adam_id = adapter._get_voice_id("Adam")
        assert adam_id == "pNInz6obpgDQGcFmaJgB"

        custom_id = "custom-voice-id-12345"
        assert adapter._get_voice_id(custom_id) == custom_id

    @pytest.mark.asyncio
    async def test_elevenlabs_speech_basic_mp3(self, mock_api_key):
        """测试 ElevenLabs 基础 MP3 语音合成"""
        config = ProviderConfig(provider_type="elevenlabs", api_key=mock_api_key)
        adapter = ElevenLabsAdapter(config=config)
        await adapter.start()

        test_audio = b"elevenlabs mp3 audio data"

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_speech_response(test_audio)

            result = await adapter.speech(
                model="eleven_multilingual_v2",
                input="Hello, this is ElevenLabs TTS test",
                voice="Rachel",
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args

            assert "/text-to-speech/21m00Tcm4TlvDq8ikWAM" in call_args.args[0]

            json_body = call_args.kwargs.get("json", {})
            assert json_body["text"] == "Hello, this is ElevenLabs TTS test"
            assert json_body["model_id"] == "eleven_multilingual_v2"

            assert isinstance(result, SpeechResult)
            assert result.audio_data == test_audio
            assert result.format == "mp3"
            assert result.content_type == "audio/mpeg"
            assert result.model == "eleven_multilingual_v2"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_elevenlabs_speech_with_voice_settings(self, mock_api_key):
        """测试带音色参数（stability/similarity/style）的合成"""
        config = ProviderConfig(provider_type="elevenlabs", api_key=mock_api_key)
        adapter = ElevenLabsAdapter(config=config)
        await adapter.start()

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_speech_response(b"audio")

            await adapter.speech(
                model="eleven_multilingual_v2",
                input="Voice settings test",
                voice="Adam",
                stability=0.7,
                similarity_boost=0.85,
                style=0.4,
                use_speaker_boost=True,
            )

            json_body = mock_post.call_args.kwargs.get("json", {})
            voice_settings = json_body.get("voice_settings", {})
            assert voice_settings["stability"] == 0.7
            assert voice_settings["similarity_boost"] == 0.85
            assert voice_settings["style"] == 0.4
            assert voice_settings["use_speaker_boost"] is True

        await adapter.close()

    @pytest.mark.asyncio
    async def test_elevenlabs_speech_multiple_formats(self, mock_api_key):
        """测试生成多种音频格式（wav/ogg）"""
        config = ProviderConfig(provider_type="elevenlabs", api_key=mock_api_key)
        adapter = ElevenLabsAdapter(config=config)
        await adapter.start()

        format_tests = [
            ("wav", "audio/wav"),
            ("ogg", "audio/ogg"),
            ("mp3", "audio/mpeg"),
        ]

        for fmt, expected_ct in format_tests:
            with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = self._mock_speech_response(b"data", expected_ct)

                result = await adapter.speech(
                    model="eleven_turbo_v2_5",
                    input=f"Format {fmt} test",
                    voice="Sarah",
                    response_format=fmt,
                )

                assert result.format == fmt
                assert result.content_type == expected_ct
                assert result.model == "eleven_turbo_v2_5"

                call_kwargs = mock_post.call_args.kwargs
                params = call_kwargs.get("params", {})
                if fmt != "mp3":
                    assert params.get("output_format") == fmt

        await adapter.close()

    @pytest.mark.asyncio
    async def test_elevenlabs_transcribe_raises_unsupported(self, mock_api_key):
        """测试 ElevenLabs 不支持 ASR 时抛出异常"""
        from agn.core.errors import UnsupportedCapabilityError

        config = ProviderConfig(provider_type="elevenlabs", api_key=mock_api_key)
        adapter = ElevenLabsAdapter(config=config)

        with pytest.raises(UnsupportedCapabilityError, match="speech-to-text"):
            await adapter.transcribe(
                model="any-model",
                file=b"audio",
            )

    @pytest.mark.asyncio
    async def test_elevenlabs_authentication_error(self, mock_api_key):
        """测试 ElevenLabs 认证错误处理"""
        from agn.core.errors import AuthenticationError

        config = ProviderConfig(provider_type="elevenlabs", api_key="invalid-key")
        adapter = ElevenLabsAdapter(config=config)
        await adapter.start()

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_error_response(401, {"detail": {"message": "Unauthorized"}})

            with pytest.raises(AuthenticationError, match="Invalid ElevenLabs API key"):
                await adapter.speech(
                    model="eleven_multilingual_v2",
                    input="test",
                    voice="Rachel",
                )

        await adapter.close()

    @pytest.mark.asyncio
    async def test_elevenlabs_voice_not_found(self, mock_api_key):
        """测试 voice_id 不存在时的 404 错误"""
        from agn.core.errors import APIError

        config = ProviderConfig(provider_type="elevenlabs", api_key=mock_api_key)
        adapter = ElevenLabsAdapter(config=config)
        await adapter.start()

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_error_response(404)

            with pytest.raises(APIError) as exc_info:
                await adapter.speech(
                    model="eleven_multilingual_v2",
                    input="test",
                    voice="nonexistent-voice",
                )
            assert exc_info.value.status_code == 404

        await adapter.close()

    @pytest.mark.asyncio
    async def test_elevenlabs_chat_raises_unsupported(self, mock_api_key):
        """测试 ElevenLabs 不支持 chat"""
        from agn.core.errors import UnsupportedCapabilityError

        config = ProviderConfig(provider_type="elevenlabs", api_key=mock_api_key)
        adapter = ElevenLabsAdapter(config=config)

        with pytest.raises(UnsupportedCapabilityError, match="chat"):
            await adapter.chat(model="test", messages=[])

    @pytest.mark.asyncio
    async def test_elevenlabs_get_voice_id_direct_id(self):
        """测试直接传入 voice_id（非内置名称）"""
        config = ProviderConfig(provider_type="elevenlabs", api_key="test")
        adapter = ElevenLabsAdapter(config=config)

        custom_id = "my-custom-voice-id-abc123"
        assert adapter._get_voice_id(custom_id) == custom_id


# ==================== 路由映射扩展测试 ====================


class TestRouterAudioMappingExtended:
    """测试扩展后的 Router 语音模型映射（Groq + ElevenLabs）"""

    def test_router_has_groq_whisper_mappings(self):
        """测试路由表包含 Groq Whisper 模型映射"""
        from agn.router import Router

        expected = {
            "whisper-large-v3": "groq",
            "whisper-large-v3-turbo": "groq",
            "distil-whisper-large-v3-en": "groq",
        }

        MODEL_MAP = Router.MODEL_PROVIDER_MAP
        for model_id, provider in expected.items():
            assert model_id in MODEL_MAP, f"路由表缺少 Groq 模型: {model_id}"
            assert MODEL_MAP[model_id] == provider, \
                f"{model_id} 应映射到 {provider}，实际是 {MODEL_MAP[model_id]}"

    def test_router_has_elevenlabs_mappings(self):
        """测试路由表包含 ElevenLabs TTS 模型映射"""
        from agn.router import Router

        expected_models = [
            "eleven_multilingual_v2",
            "eleven_multilingual_v1",
            "eleven_monolingual_v1",
            "eleven_turbo_v2_5",
            "eleven_turbo_v2",
            "eleven_flash_v2_5",
        ]

        MODEL_MAP = Router.MODEL_PROVIDER_MAP
        for model_id in expected_models:
            assert model_id in MODEL_MAP, f"路由表缺少 ElevenLabs 模型: {model_id}"
            assert MODEL_MAP[model_id] == "elevenlabs", \
                f"{model_id} 应映射到 elevenlabs，实际是 {MODEL_MAP[model_id]}"

    def test_whisper_large_v3_routes_to_groq_not_openai(self):
        """测试 whisper-large-v3（Groq）与 whisper-1（OpenAI）不冲突"""
        from agn.router import Router

        MODEL_MAP = Router.MODEL_PROVIDER_MAP
        assert MODEL_MAP["whisper-1"] == "openai"
        assert MODEL_MAP["whisper-large-v3"] == "groq"
        assert MODEL_MAP["whisper-large-v3-turbo"] == "groq"

    def test_elevenlabs_adapter_factory_registration(self):
        """测试 ElevenLabs 适配器已在工厂注册"""
        from agn.adapters import ElevenLabsAdapter
        from agn.adapters.factory import AdapterFactory

        assert AdapterFactory.get_adapter_class("elevenlabs") is ElevenLabsAdapter
        assert AdapterFactory.get_adapter_class("eleven") is ElevenLabsAdapter
        assert AdapterFactory.get_adapter_class("11labs") is ElevenLabsAdapter


# ==================== Deepgram ASR 专用测试 ====================


class TestDeepgramGetAudioBytes:
    """测试 Deepgram 自己的 get_audio_bytes 方法（返回 (bytes, mime_type)）"""

    @pytest.mark.asyncio
    async def test_deepgram_get_audio_bytes_from_bytes(self, mock_api_key):
        """测试 bytes 输入返回正确 MIME 类型"""
        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)

        test_data = b"raw wav data"
        data, mime = await adapter.get_audio_bytes(test_data)
        assert data == test_data
        assert mime == "audio/wav"

    @pytest.mark.asyncio
    async def test_deepgram_get_audio_bytes_from_mp3_path(self, mock_api_key, tmp_path):
        """测试 MP3 文件路径返回 audio/mpeg MIME"""
        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)

        test_data = b"mp3 content"
        test_file = tmp_path / "audio.mp3"
        test_file.write_bytes(test_data)

        data, mime = await adapter.get_audio_bytes(str(test_file))
        assert data == test_data
        assert mime == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_deepgram_get_audio_bytes_from_flac_path(self, mock_api_key, tmp_path):
        """测试 FLAC 文件路径返回 audio/flac MIME"""
        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)

        test_data = b"flac content"
        test_file = tmp_path / "audio.flac"
        test_file.write_bytes(test_data)

        data, mime = await adapter.get_audio_bytes(str(test_file))
        assert data == test_data
        assert mime == "audio/flac"

    @pytest.mark.asyncio
    async def test_deepgram_get_audio_bytes_from_ogg_path(self, mock_api_key, tmp_path):
        """测试 OGG 文件路径"""
        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)

        test_data = b"ogg content"
        test_file = tmp_path / "audio.ogg"
        test_file.write_bytes(test_data)

        data, mime = await adapter.get_audio_bytes(str(test_file))
        assert data == test_data
        assert mime == "audio/ogg"

    @pytest.mark.asyncio
    async def test_deepgram_get_audio_bytes_from_file_object(self, mock_api_key):
        """测试文件对象输入"""
        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)

        test_data = b"file obj data"
        file_obj = io.BytesIO(test_data)
        file_obj.name = "recording.mp3"

        data, mime = await adapter.get_audio_bytes(file_obj)
        assert data == test_data
        assert mime == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_deepgram_get_audio_bytes_from_base64(self, mock_api_key):
        """测试纯 base64 输入"""
        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)

        original = b"base64 audio content"
        b64 = base64.b64encode(original).decode("utf-8")
        data, mime = await adapter.get_audio_bytes(b64)
        assert data == original
        assert mime == "audio/wav"

    @pytest.mark.asyncio
    async def test_deepgram_get_audio_bytes_from_data_uri_mp3(self, mock_api_key):
        """测试 data:audio/mpeg;base64, 前缀的 base64"""
        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)

        original = b"mp3 base64 data"
        b64 = base64.b64encode(original).decode("utf-8")
        data_uri = f"data:audio/mp3;base64,{b64}"
        data, mime = await adapter.get_audio_bytes(data_uri)
        assert data == original
        assert mime == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_deepgram_get_audio_bytes_unsupported_type(self, mock_api_key):
        """测试不支持的类型抛出 ValueError"""
        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)

        with pytest.raises(ValueError, match="Unsupported file input type"):
            await adapter.get_audio_bytes(42)

    @pytest.mark.asyncio
    async def test_deepgram_get_audio_bytes_nonexistent_path(self, mock_api_key):
        """测试不存在的路径抛出异常"""
        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)

        with pytest.raises((FileNotFoundError, OSError)):
            await adapter.get_audio_bytes("/nonexistent/audio.wav")


class TestDeepgramASR:
    """测试 Deepgram Nova 系列高速 ASR"""

    def _mock_response(self, json_data, status_code=200):
        """创建模拟 HTTP 响应"""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json = MagicMock(return_value=json_data)
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    def _make_deepgram_response(
        self,
        transcript: str,
        duration: float = 3.5,
        language: str = "en",
        words: list[dict] | None = None,
        paragraphs: list[dict] | None = None,
        model: str = "nova-2",
    ) -> dict:
        """构造标准 Deepgram 响应"""
        response = {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": transcript,
                                "confidence": 0.98,
                            }
                        ]
                    }
                ]
            },
            "metadata": {
                "duration": duration,
                "model_info": {
                    "name": model,
                    "language": language,
                },
            },
        }

        if words:
            response["results"]["channels"][0]["alternatives"][0]["words"] = words
        if paragraphs:
            response["results"]["channels"][0]["alternatives"][0]["paragraphs"] = {
                "paragraphs": paragraphs
            }

        return response

    @pytest.mark.asyncio
    async def test_deepgram_transcribe_basic_nova2(self, mock_api_key):
        """测试 Deepgram Nova-2 基础转写"""
        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)
        await adapter.start()

        mock_result = self._make_deepgram_response(
            transcript="Hello, this is Deepgram Nova 2 test",
            duration=2.5,
            language="en",
        )

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.transcribe(
                model="nova-2",
                file=b"fake audio bytes",
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args

            assert "/listen" in call_args.args[0]

            params = call_args.kwargs.get("params", {})
            assert params["model"] == "nova-2"
            assert params.get("smart_format") == "true"
            assert params.get("punctuate") == "true"

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, this is Deepgram Nova 2 test"
            assert result.language == "en"
            assert result.duration == 2.5
            assert result.model == "nova-2"
            assert result.task == "transcribe"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_deepgram_transcribe_with_words(self, mock_api_key):
        """测试带词级时间戳的转写结果解析"""
        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)
        await adapter.start()

        words = [
            {"word": "Hello", "start": 0.0, "end": 0.4, "confidence": 0.99},
            {"word": "world", "start": 0.4, "end": 0.8, "confidence": 0.97},
            {"word": "test", "start": 0.8, "end": 1.2, "confidence": 0.95},
        ]

        mock_result = self._make_deepgram_response(
            transcript="Hello world test",
            words=words,
        )

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.transcribe(
                model="nova-2",
                file=b"audio",
            )

            assert result.words is not None
            assert len(result.words) == 3
            assert result.words[0].word == "Hello"
            assert result.words[0].start == 0.0
            assert result.words[0].end == 0.4
            assert result.words[0].confidence == 0.99
            assert result.words[2].word == "test"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_deepgram_transcribe_with_diarization(self, mock_api_key):
        """测试说话人分离（diarize）参数传递和 speaker 解析"""
        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)
        await adapter.start()

        words = [
            {"word": "Hello", "start": 0.0, "end": 0.3, "speaker": 0},
            {"word": "world", "start": 0.3, "end": 0.6, "speaker": 0},
            {"word": "Hi", "start": 0.8, "end": 1.0, "speaker": 1},
            {"word": "there", "start": 1.0, "end": 1.3, "speaker": 1},
        ]

        paragraphs = [
            {
                "speaker": 0,
                "sentences": [
                    {"text": "Hello world", "start": 0.0, "end": 0.6},
                ],
            },
            {
                "speaker": 1,
                "sentences": [
                    {"text": "Hi there", "start": 0.8, "end": 1.3},
                ],
            },
        ]

        mock_result = self._make_deepgram_response(
            transcript="Hello world Hi there",
            words=words,
            paragraphs=paragraphs,
        )

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.transcribe(
                model="nova-2",
                file=b"audio",
                diarize=True,
            )

            call_kwargs = mock_post.call_args.kwargs
            params = call_kwargs.get("params", {})
            assert params.get("diarize") == "true"

            assert result.words is not None
            assert result.words[0].speaker == 0
            assert result.words[2].speaker == 1

            assert result.segments is not None
            assert len(result.segments) == 2
            assert result.segments[0].text == "Hello world"
            assert result.segments[0].speaker == 0
            assert result.segments[1].text == "Hi there"
            assert result.segments[1].speaker == 1

        await adapter.close()

    @pytest.mark.asyncio
    async def test_deepgram_transcribe_with_language(self, mock_api_key):
        """测试指定语言参数传递"""
        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)
        await adapter.start()

        mock_result = self._make_deepgram_response(
            transcript="你好，这是中文测试",
            language="zh",
            duration=2.0,
        )

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.transcribe(
                model="nova-2",
                file=b"chinese audio",
                language="zh",
            )

            params = mock_post.call_args.kwargs.get("params", {})
            assert params["language"] == "zh"
            assert result.text == "你好，这是中文测试"
            assert result.language == "zh"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_deepgram_transcribe_content_type_from_ext(self, mock_api_key, tmp_path):
        """测试文件扩展名决定 Content-Type"""
        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)
        await adapter.start()

        test_file = tmp_path / "audio.flac"
        test_file.write_bytes(b"flac audio")

        mock_result = self._make_deepgram_response("ok")

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            await adapter.transcribe(model="nova-3", file=str(test_file))

            call_kwargs = mock_post.call_args.kwargs
            headers = call_kwargs.get("headers", {})
            assert headers.get("Content-Type") == "audio/flac"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_deepgram_speech_raises_unsupported(self, mock_api_key):
        """测试 Deepgram 不支持 TTS 时抛出异常"""
        from agn.core.errors import UnsupportedCapabilityError

        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)

        with pytest.raises(UnsupportedCapabilityError, match="speech"):
            await adapter.speech(model="any", input="test", voice="x")

    @pytest.mark.asyncio
    async def test_deepgram_authentication_error(self, mock_api_key):
        """测试 Deepgram 401 认证错误"""
        from agn.core.errors import AuthenticationError

        config = ProviderConfig(provider_type="deepgram", api_key="bad-key")
        adapter = DeepgramAdapter(config=config)
        await adapter.start()

        error_resp = MagicMock()
        error_resp.status_code = 401
        error_resp.json = MagicMock(return_value={"err_msg": "Invalid API key"})
        error_resp.text = "Unauthorized"

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = error_resp

            with pytest.raises(AuthenticationError, match="Invalid Deepgram"):
                await adapter.transcribe(model="nova-2", file=b"audio")

        await adapter.close()

    @pytest.mark.asyncio
    async def test_deepgram_rate_limit_error(self, mock_api_key):
        """测试 Deepgram 429 限流错误"""
        from agn.core.errors import RateLimitError

        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)
        await adapter.start()

        error_resp = MagicMock()
        error_resp.status_code = 429
        error_resp.json = MagicMock(return_value={"err_msg": "Rate limit exceeded"})

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = error_resp

            with pytest.raises(RateLimitError, match="Deepgram rate limit"):
                await adapter.transcribe(model="nova-2", file=b"audio")

        await adapter.close()

    @pytest.mark.asyncio
    async def test_deepgram_bad_request_error(self, mock_api_key):
        """测试 Deepgram 400 参数错误"""
        from agn.core.errors import APIError

        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)
        await adapter.start()

        error_resp = MagicMock()
        error_resp.status_code = 400
        error_resp.json = MagicMock(return_value={"err_msg": "Invalid model parameter"})
        error_resp.text = '{"err_msg":"Invalid model parameter"}'

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = error_resp

            with pytest.raises(APIError) as exc_info:
                await adapter.transcribe(model="invalid-model", file=b"audio")
            assert exc_info.value.status_code == 400

        await adapter.close()

    @pytest.mark.asyncio
    async def test_deepgram_nova3_transcribe(self, mock_api_key):
        """测试 Deepgram Nova-3 最新模型转写"""
        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)
        await adapter.start()

        mock_result = self._make_deepgram_response(
            transcript="Nova 3 accuracy test",
            model="nova-3",
        )

        with patch.object(adapter._http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = self._mock_response(mock_result)

            result = await adapter.transcribe(
                model="nova-3",
                file=b"audio",
            )

            params = mock_post.call_args.kwargs.get("params", {})
            assert params["model"] == "nova-3"
            assert result.model == "nova-3"
            assert result.text == "Nova 3 accuracy test"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_deepgram_chat_raises_unsupported(self, mock_api_key):
        """测试 Deepgram 不支持 chat"""
        from agn.core.errors import UnsupportedCapabilityError

        config = ProviderConfig(provider_type="deepgram", api_key=mock_api_key)
        adapter = DeepgramAdapter(config=config)

        with pytest.raises(UnsupportedCapabilityError, match="chat"):
            await adapter.chat(model="test", messages=[])

    @pytest.mark.asyncio
    async def test_deepgram_not_started_raises(self):
        """测试未调用 start() 时调用方法抛出 RuntimeError"""
        config = ProviderConfig(provider_type="deepgram", api_key="test")
        adapter = DeepgramAdapter(config=config)

        with pytest.raises(RuntimeError, match="not started"):
            await adapter.transcribe(model="nova-2", file=b"audio")


# ==================== 路由映射 Deepgram 测试 ====================


class TestRouterDeepgramMapping:
    """测试 Router 中 Deepgram 模型映射"""

    def test_router_has_deepgram_nova_models(self):
        """测试路由表包含 Deepgram Nova 系列模型映射"""
        from agn.router import Router

        expected = {
            "nova-3": "deepgram",
            "nova-2": "deepgram",
            "nova-2-general": "deepgram",
            "nova-2-meeting": "deepgram",
            "nova-2-phonecall": "deepgram",
            "nova-2-conversationalai": "deepgram",
            "nova-2-video": "deepgram",
            "nova-2-medical": "deepgram",
            "nova-2-finance": "deepgram",
            "nova-2-drivethru": "deepgram",
        }

        MODEL_MAP = Router.MODEL_PROVIDER_MAP
        for model_id, provider in expected.items():
            assert model_id in MODEL_MAP, f"路由表缺少 Deepgram 模型: {model_id}"
            assert MODEL_MAP[model_id] == provider, \
                f"{model_id} 应映射到 {provider}，实际是 {MODEL_MAP[model_id]}"

    def test_router_has_deepgram_whisper_models(self):
        """测试路由表包含 Deepgram 托管的 Whisper 模型"""
        from agn.router import Router

        MODEL_MAP = Router.MODEL_PROVIDER_MAP
        assert MODEL_MAP.get("whisper-large") == "deepgram"
        assert MODEL_MAP.get("whisper-medium") == "deepgram"
        assert MODEL_MAP.get("whisper-small") == "deepgram"

    def test_router_has_deepgram_legacy_models(self):
        """测试路由表包含 Deepgram 旧版模型"""
        from agn.router import Router

        MODEL_MAP = Router.MODEL_PROVIDER_MAP
        assert MODEL_MAP.get("enhanced") == "deepgram"
        assert MODEL_MAP.get("base") == "deepgram"

    def test_deepgram_adapter_factory_registration(self):
        """测试 Deepgram 适配器已在工厂注册"""
        from agn.adapters import DeepgramAdapter
        from agn.adapters.factory import AdapterFactory

        assert AdapterFactory.get_adapter_class("deepgram") is DeepgramAdapter
        assert AdapterFactory.get_adapter_class("dg") is DeepgramAdapter

    def test_whisper_model_routing_no_conflict(self):
        """测试各平台 Whisper 模型路由不冲突"""
        from agn.router import Router

        MODEL_MAP = Router.MODEL_PROVIDER_MAP
        assert MODEL_MAP["whisper-1"] == "openai"
        assert MODEL_MAP["whisper-large-v3"] == "groq"
        assert MODEL_MAP["whisper-large-v3-turbo"] == "groq"
        assert MODEL_MAP["whisper-large"] == "deepgram"
        assert MODEL_MAP["openai/whisper-large-v3"] == "togetherai"


# ==================== AssemblyAI 企业级 ASR 单元测试 ====================


class TestAssemblyAIGetAudioBytes:
    """测试 AssemblyAI 音频输入处理（字节/路径/base64/文件对象）"""

    def setup_method(self):
        """每个测试方法前初始化适配器"""
        config = ProviderConfig(provider_type="assemblyai", api_key="test-key")
        self.adapter = AssemblyAIAdapter(config=config)

    def test_get_audio_bytes_with_raw_bytes(self):
        """测试 bytes 输入直接返回"""
        import pytest
        pytestmark = pytest.mark.asyncio

        async def _test():
            await self.adapter.start()
            try:
                data, mime = await self.adapter.get_audio_bytes(b"fake audio data")
                assert data == b"fake audio data"
                assert mime == "application/octet-stream"
            finally:
                await self.adapter.close()

        import asyncio
        asyncio.run(_test())

    def test_get_audio_bytes_with_file_path(self):
        """测试从磁盘文件读取音频"""
        async def _test():
            await self.adapter.start()
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(b"fake mp3 bytes")
                    tmp_path = f.name
                try:
                    data, mime = await self.adapter.get_audio_bytes(tmp_path)
                    assert data == b"fake mp3 bytes"
                    assert mime == "application/octet-stream"
                finally:
                    import os
                    os.unlink(tmp_path)
            finally:
                await self.adapter.close()

        import asyncio
        asyncio.run(_test())

    def test_get_audio_bytes_with_nonexistent_path_raises(self):
        """测试不存在的文件路径抛出 FileNotFoundError"""
        async def _test():
            await self.adapter.start()
            try:
                with pytest.raises(FileNotFoundError):
                    await self.adapter.get_audio_bytes("/nonexistent/audio.wav")
            finally:
                await self.adapter.close()

        import asyncio
        asyncio.run(_test())

    def test_get_audio_bytes_with_file_object(self):
        """测试类文件对象读取"""
        async def _test():
            await self.adapter.start()
            try:
                fobj = io.BytesIO(b"audio from buffer")
                data, mime = await self.adapter.get_audio_bytes(fobj)
                assert data == b"audio from buffer"
                assert mime == "application/octet-stream"
            finally:
                await self.adapter.close()

        import asyncio
        asyncio.run(_test())

    def test_get_audio_bytes_with_data_uri(self):
        """测试 data:audio/...;base64 URI 解码"""
        import base64 as b64
        async def _test():
            await self.adapter.start()
            try:
                payload = b64.b64encode(b"hello audio").decode()
                data_uri = f"data:audio/wav;base64,{payload}"
                data, mime = await self.adapter.get_audio_bytes(data_uri)
                assert data == b"hello audio"
                assert mime == "application/octet-stream"
            finally:
                await self.adapter.close()

        import asyncio
        asyncio.run(_test())

    def test_get_audio_bytes_with_invalid_type_raises(self):
        """测试不支持的输入类型抛出 ValueError"""
        async def _test():
            await self.adapter.start()
            try:
                with pytest.raises(ValueError, match="Unsupported file input type"):
                    await self.adapter.get_audio_bytes(12345)
            finally:
                await self.adapter.close()

        import asyncio
        asyncio.run(_test())


class TestAssemblyAIASR:
    """测试 AssemblyAI 语音转文字"""

    def setup_method(self):
        self.config = ProviderConfig(provider_type="assemblyai", api_key="test-aai-key")
        self.adapter = AssemblyAIAdapter(config=self.config)

    def test_adapter_capabilities(self):
        """测试 AssemblyAI 仅支持 ASR"""
        assert Capabilities.AUDIO_TRANSCRIBE in self.adapter.supported_capabilities
        assert Capabilities.AUDIO_SPEECH not in self.adapter.supported_capabilities
        assert self.adapter.provider_type == "assemblyai"

    @pytest.mark.asyncio
    async def test_not_started_raises(self):
        """测试未启动时调用 transcribe 抛出异常"""
        with pytest.raises(RuntimeError, match="Adapter not started"):
            await self.adapter.transcribe("best", b"fake audio")

    @pytest.mark.asyncio
    async def test_speech_raises_unsupported(self, mock_api_key):
        """测试 TTS 能力不支持"""
        from agn.core.errors import UnsupportedCapabilityError
        await self.adapter.start()
        try:
            with pytest.raises(UnsupportedCapabilityError):
                await self.adapter.speech("best", "hello")
        finally:
            await self.adapter.close()

    @pytest.mark.asyncio
    async def test_transcribe_basic_flow(self):
        """测试基本 ASR 流程：上传 -> 提交 -> 轮询 -> 解析"""
        await self.adapter.start()
        try:
            mock_client = AsyncMock()
            upload_resp = MagicMock()
            upload_resp.status_code = 200
            upload_resp.json.return_value = {"upload_url": "https://cdn.assemblyai.com/upload/abc123"}

            submit_resp = MagicMock()
            submit_resp.status_code = 200
            submit_resp.json.return_value = {"id": "trans-123", "status": "queued"}

            poll_processing = MagicMock()
            poll_processing.status_code = 200
            poll_processing.json.return_value = {"id": "trans-123", "status": "processing"}

            poll_completed = MagicMock()
            poll_completed.status_code = 200
            poll_completed.json.return_value = {
                "id": "trans-123",
                "status": "completed",
                "text": "Hello world, this is a test.",
                "language_code": "en",
                "audio_duration": 2.5,
                "words": [
                    {"text": "Hello", "start": 100, "end": 300, "confidence": 0.99},
                    {"text": "world", "start": 350, "end": 600, "confidence": 0.97},
                ],
                "confidence": 0.98,
            }

            mock_client.post = AsyncMock(side_effect=[upload_resp, submit_resp])
            mock_client.get = AsyncMock(side_effect=[poll_processing, poll_completed])

            self.adapter._http_client = mock_client
            self.adapter._handle_error = MagicMock()

            result = await self.adapter.transcribe(
                "best",
                b"fake audio data",
                polling_interval=0.001,
            )

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello world, this is a test."
            assert result.language == "en"
            assert result.duration == 2.5
            assert result.words is not None
            assert len(result.words) == 2
            assert result.words[0].word == "Hello"
            assert result.words[0].start == pytest.approx(0.1)
            assert result.words[0].confidence == pytest.approx(0.99)
        finally:
            await self.adapter.close()

    @pytest.mark.asyncio
    async def test_transcribe_with_speaker_labels(self):
        """测试说话人分离（utterances）场景"""
        await self.adapter.start()
        try:
            mock_client = AsyncMock()

            upload_resp = MagicMock(status_code=200)
            upload_resp.json.return_value = {"upload_url": "https://cdn.assemblyai.com/upload/abc"}

            submit_resp = MagicMock(status_code=200)
            submit_resp.json.return_value = {"id": "t1", "status": "queued"}

            completed = MagicMock(status_code=200)
            completed.json.return_value = {
                "id": "t1",
                "status": "completed",
                "text": "Hello from speaker A and speaker B.",
                "language_code": "en",
                "utterances": [
                    {
                        "start": 0,
                        "end": 1500,
                        "text": "Hello from speaker A",
                        "confidence": 0.98,
                        "speaker": "A",
                        "words": [
                            {"text": "Hello", "start": 0, "end": 500, "confidence": 0.99, "speaker": "A"},
                            {"text": "from", "start": 500, "end": 1000, "confidence": 0.97, "speaker": "A"},
                        ],
                    },
                    {
                        "start": 1500,
                        "end": 3000,
                        "text": "and speaker B",
                        "confidence": 0.96,
                        "speaker": "B",
                        "words": [
                            {"text": "and", "start": 1500, "end": 2000, "confidence": 0.95, "speaker": "B"},
                        ],
                    },
                ],
            }

            mock_client.post = AsyncMock(side_effect=[upload_resp, submit_resp])
            mock_client.get = AsyncMock(return_value=completed)
            self.adapter._http_client = mock_client
            self.adapter._handle_error = MagicMock()

            result = await self.adapter.transcribe(
                "best", b"audio", speaker_labels=True, polling_interval=0.001,
            )

            assert result.segments is not None
            assert len(result.segments) == 2
            assert result.segments[0].speaker == "A"
            assert result.segments[1].speaker == "B"
            assert result.words is not None
            assert len(result.words) == 3
            assert result.words[0].speaker == "A"
            assert result.words[2].speaker == "B"
        finally:
            await self.adapter.close()

    @pytest.mark.asyncio
    async def test_transcribe_uses_audio_url_directly(self):
        """测试传入 audio_url 时跳过上传"""
        await self.adapter.start()
        try:
            mock_client = AsyncMock()
            submit_resp = MagicMock(status_code=200)
            submit_resp.json.return_value = {"id": "t-url", "status": "queued"}
            completed = MagicMock(status_code=200)
            completed.json.return_value = {
                "id": "t-url",
                "status": "completed",
                "text": "direct url test",
                "words": [],
            }
            mock_client.post = AsyncMock(return_value=submit_resp)
            mock_client.get = AsyncMock(return_value=completed)
            self.adapter._http_client = mock_client
            self.adapter._handle_error = MagicMock()

            result = await self.adapter.transcribe(
                "best", b"x",
                audio_url="https://example.com/audio.mp3",
                polling_interval=0.001,
            )
            assert result.text == "direct url test"
            assert mock_client.post.call_count == 1
            submitted_payload = mock_client.post.call_args_list[0]
            assert submitted_payload.kwargs["json"]["audio_url"] == "https://example.com/audio.mp3"
        finally:
            await self.adapter.close()

    @pytest.mark.asyncio
    async def test_transcribe_passes_language_and_options(self):
        """测试语言代码和其他参数传递"""
        await self.adapter.start()
        try:
            mock_client = AsyncMock()
            upload_resp = MagicMock(status_code=200)
            upload_resp.json.return_value = {"upload_url": "https://u"}
            submit_resp = MagicMock(status_code=200)
            submit_resp.json.return_value = {"id": "t2", "status": "queued"}
            completed = MagicMock(status_code=200)
            completed.json.return_value = {
                "id": "t2", "status": "completed", "text": "测试", "words": [],
            }
            mock_client.post = AsyncMock(side_effect=[upload_resp, submit_resp])
            mock_client.get = AsyncMock(return_value=completed)
            self.adapter._http_client = mock_client
            self.adapter._handle_error = MagicMock()

            await self.adapter.transcribe(
                "best", b"audio",
                language_code="zh",
                filter_profanity=True,
                sentiment_analysis=True,
                auto_chapters=True,
                entity_detection=True,
                word_boost=["custom term"],
                polling_interval=0.001,
            )

            submitted_json = mock_client.post.call_args_list[1].kwargs["json"]
            assert submitted_json["language_code"] == "zh"
            assert submitted_json["filter_profanity"] is True
            assert submitted_json["sentiment_analysis"] is True
            assert submitted_json["auto_chapters"] is True
            assert submitted_json["entity_detection"] is True
            assert submitted_json["word_boost"] == ["custom term"]
        finally:
            await self.adapter.close()

    @pytest.mark.asyncio
    async def test_transcribe_defaults_model_to_best(self):
        """测试非法模型名默认回退到 best"""
        await self.adapter.start()
        try:
            mock_client = AsyncMock()
            upload_resp = MagicMock(status_code=200)
            upload_resp.json.return_value = {"upload_url": "https://u"}
            submit_resp = MagicMock(status_code=200)
            submit_resp.json.return_value = {"id": "t3", "status": "queued"}
            completed = MagicMock(status_code=200)
            completed.json.return_value = {
                "id": "t3", "status": "completed", "text": "ok", "words": [],
            }
            mock_client.post = AsyncMock(side_effect=[upload_resp, submit_resp])
            mock_client.get = AsyncMock(return_value=completed)
            self.adapter._http_client = mock_client
            self.adapter._handle_error = MagicMock()

            await self.adapter.transcribe(
                "invalid_model", b"audio", polling_interval=0.001,
            )
            submitted_json = mock_client.post.call_args_list[1].kwargs["json"]
            assert submitted_json["speech_model"] == "best"
        finally:
            await self.adapter.close()

    @pytest.mark.asyncio
    async def test_transcribe_handles_error_status(self):
        """测试转写错误状态抛出异常"""
        from agn.core.errors import APIError
        await self.adapter.start()
        try:
            mock_client = AsyncMock()
            upload_resp = MagicMock(status_code=200)
            upload_resp.json.return_value = {"upload_url": "https://u"}
            submit_resp = MagicMock(status_code=200)
            submit_resp.json.return_value = {"id": "t-err", "status": "queued"}
            err_poll = MagicMock(status_code=200)
            err_poll.json.return_value = {
                "id": "t-err", "status": "error", "error": "Audio too short",
            }
            mock_client.post = AsyncMock(side_effect=[upload_resp, submit_resp])
            mock_client.get = AsyncMock(return_value=err_poll)
            self.adapter._http_client = mock_client
            self.adapter._handle_error = MagicMock()

            with pytest.raises(APIError, match="Audio too short"):
                await self.adapter.transcribe("best", b"x", polling_interval=0.001, max_polls=5)
        finally:
            await self.adapter.close()

    @pytest.mark.asyncio
    async def test_list_models(self):
        """测试列出 AssemblyAI 模型"""
        await self.adapter.start()
        try:
            models = await self.adapter.list_models()
            assert len(models) == 2
            ids = {m.id for m in models}
            assert "best" in ids
            assert "nano" in ids
            for m in models:
                assert m.provider == "assemblyai"
                assert "audio_transcribe" in m.capabilities
        finally:
            await self.adapter.close()

    def test_error_responses(self):
        """测试 HTTP 错误处理"""
        from agn.core.errors import APIError, AuthenticationError, RateLimitError
        # 401
        resp = MagicMock(status_code=401)
        resp.json.return_value = {"error": "Unauthorized"}
        with pytest.raises(AuthenticationError):
            self.adapter._handle_error(resp)

        # 429
        resp.status_code = 429
        resp.json.return_value = {"error": "Rate limit"}
        with pytest.raises(RateLimitError):
            self.adapter._handle_error(resp)

        # 500
        resp.status_code = 500
        resp.json.return_value = {"error": "Server error"}
        with pytest.raises(APIError):
            self.adapter._handle_error(resp)


# ==================== Cartesia Sonic TTS 单元测试 ====================


class TestCartesiaTTS:
    """测试 Cartesia Sonic 文字转语音"""

    def setup_method(self):
        self.config = ProviderConfig(provider_type="cartesia", api_key="test-cart-key")
        self.adapter = CartesiaAdapter(config=self.config)

    def test_adapter_capabilities(self):
        """测试 Cartesia 仅支持 TTS"""
        assert Capabilities.AUDIO_SPEECH in self.adapter.supported_capabilities
        assert Capabilities.AUDIO_TRANSCRIBE not in self.adapter.supported_capabilities
        assert self.adapter.provider_type == "cartesia"

    def test_default_voices_preset(self):
        """测试内置音色名称到ID映射"""
        voices = self.adapter.DEFAULT_VOICES
        assert "Generic Woman" in voices
        assert "Generic Man" in voices
        assert "Chinese Woman" in voices
        assert "The Don" in voices
        assert len(voices["Generic Woman"]) == 36

    def test_get_voice_id_resolves_name(self):
        """测试音色名自动解析为ID"""
        vid = self.adapter._get_voice_id("Chinese Woman")
        assert vid == self.adapter.DEFAULT_VOICES["Chinese Woman"]

    def test_get_voice_id_passes_uuid(self):
        """测试直接传入UUID时原样返回"""
        uuid = "b2b21a4e-7f85-4905-a50e-70c04fb7cd9e"
        assert self.adapter._get_voice_id(uuid) == uuid

    @pytest.mark.asyncio
    async def test_not_started_raises(self):
        """测试未启动时调用 speech 抛出异常"""
        with pytest.raises(RuntimeError, match="Adapter not started"):
            await self.adapter.speech("sonic-2", "hello")

    @pytest.mark.asyncio
    async def test_transcribe_raises_unsupported(self):
        """测试 ASR 能力不支持"""
        from agn.core.errors import UnsupportedCapabilityError
        await self.adapter.start()
        try:
            with pytest.raises(UnsupportedCapabilityError):
                await self.adapter.transcribe("sonic-2", b"x")
        finally:
            await self.adapter.close()

    @pytest.mark.asyncio
    async def test_speech_basic_mp3(self):
        """测试基本 TTS 调用，返回 mp3 音频"""
        await self.adapter.start()
        try:
            mock_client = AsyncMock()
            mock_resp = MagicMock(status_code=200)
            mock_resp.content = b"FAKE_MP3_AUDIO_DATA"
            mock_client.post = AsyncMock(return_value=mock_resp)
            self.adapter._http_client = mock_client
            self.adapter._handle_error = MagicMock()

            result = await self.adapter.speech(
                "sonic-2",
                "Hello, this is a test.",
                voice="Generic Man",
            )

            assert isinstance(result, SpeechResult)
            assert result.audio_data == b"FAKE_MP3_AUDIO_DATA"
            assert result.format == "mp3"
            assert result.content_type == "audio/mpeg"
            assert result.model == "sonic-2"

            call_args = mock_client.post.call_args
            assert call_args.args[0] == "/v1/tts/bytes"
            payload = call_args.kwargs["json"]
            assert payload["model_id"] == "sonic-2"
            assert payload["transcript"] == "Hello, this is a test."
            assert payload["voice"]["mode"] == "id"
            assert payload["voice"]["id"] == self.adapter.DEFAULT_VOICES["Generic Man"]
            assert payload["output_format"]["container"] == "mp3"
        finally:
            await self.adapter.close()

    @pytest.mark.asyncio
    async def test_speech_wav_format(self):
        """测试自定义 WAV 输出格式"""
        await self.adapter.start()
        try:
            mock_client = AsyncMock()
            mock_resp = MagicMock(status_code=200)
            mock_resp.content = b"FAKE_WAV"
            mock_client.post = AsyncMock(return_value=mock_resp)
            self.adapter._http_client = mock_client
            self.adapter._handle_error = MagicMock()

            wav_fmt = {"container": "wav", "encoding": "pcm_f32le", "sample_rate": 24000}
            result = await self.adapter.speech(
                "sonic-turbo", "Testing", voice="News Lady",
                output_format=wav_fmt,
            )
            assert result.format == "wav"
            assert result.content_type == "audio/wav"

            call_kwargs = mock_client.post.call_args.kwargs
            assert call_kwargs["headers"]["Accept"] == "audio/wav"
            assert call_kwargs["json"]["output_format"] == wav_fmt
        finally:
            await self.adapter.close()

    @pytest.mark.asyncio
    async def test_speech_with_language_speed_emotion(self):
        """测试语言、语速、情感参数传递"""
        await self.adapter.start()
        try:
            mock_client = AsyncMock()
            mock_resp = MagicMock(status_code=200)
            mock_resp.content = b"audio"
            mock_client.post = AsyncMock(return_value=mock_resp)
            self.adapter._http_client = mock_client
            self.adapter._handle_error = MagicMock()

            await self.adapter.speech(
                "sonic-2", "你好世界",
                voice="Chinese Woman",
                language="zh",
                speed=1.2,
                emotion=["positivity", "curiosity"],
            )

            payload = mock_client.post.call_args.kwargs["json"]
            assert payload["language"] == "zh"
            assert payload["speed"] == 1.2
            assert payload["emotions"] == ["positivity", "curiosity"]
        finally:
            await self.adapter.close()

    @pytest.mark.asyncio
    async def test_speech_voice_id_override(self):
        """测试 voice_id 参数直接覆盖 voice 参数"""
        await self.adapter.start()
        try:
            mock_client = AsyncMock()
            mock_resp = MagicMock(status_code=200)
            mock_resp.content = b"audio"
            mock_client.post = AsyncMock(return_value=mock_resp)
            self.adapter._http_client = mock_client
            self.adapter._handle_error = MagicMock()

            custom_id = "00000000-0000-0000-0000-000000000000"
            await self.adapter.speech(
                "sonic-2", "test",
                voice="will be overridden",
                voice_id=custom_id,
            )

            payload = mock_client.post.call_args.kwargs["json"]
            assert payload["voice"]["id"] == custom_id
            assert payload["voice"]["mode"] == "id"
        finally:
            await self.adapter.close()

    @pytest.mark.asyncio
    async def test_speech_voice_embedding_clone(self):
        """测试声音克隆（voice_embedding）"""
        await self.adapter.start()
        try:
            mock_client = AsyncMock()
            mock_resp = MagicMock(status_code=200)
            mock_resp.content = b"audio"
            mock_client.post = AsyncMock(return_value=mock_resp)
            self.adapter._http_client = mock_client
            self.adapter._handle_error = MagicMock()

            embedding = [0.1, 0.2, 0.3]
            await self.adapter.speech(
                "sonic-2", "clone me",
                voice_embedding=embedding,
            )

            payload = mock_client.post.call_args.kwargs["json"]
            assert payload["voice"]["mode"] == "embedding"
            assert payload["voice"]["embedding"] == embedding
        finally:
            await self.adapter.close()

    @pytest.mark.asyncio
    async def test_speech_continuation_flag(self):
        """测试流式续写标志 continue 和 add_timestamps"""
        await self.adapter.start()
        try:
            mock_client = AsyncMock()
            mock_resp = MagicMock(status_code=200)
            mock_resp.content = b"audio"
            mock_client.post = AsyncMock(return_value=mock_resp)
            self.adapter._http_client = mock_client
            self.adapter._handle_error = MagicMock()

            await self.adapter.speech(
                "sonic-2", "continuation",
                continuation=True,
                add_timestamps=True,
            )

            payload = mock_client.post.call_args.kwargs["json"]
            assert payload["continue"] is True
            assert payload["add_timestamps"] is True
        finally:
            await self.adapter.close()

    @pytest.mark.asyncio
    async def test_list_models(self):
        """测试列出 Cartesia TTS 模型"""
        await self.adapter.start()
        try:
            models = await self.adapter.list_models()
            ids = {m.id for m in models}
            assert "sonic-2" in ids
            assert "sonic-turbo" in ids
            assert "sonic-2-2025-04-01" in ids
            for m in models:
                assert m.provider == "cartesia"
                assert "audio_speech" in m.capabilities
                assert m.type == "audio"
        finally:
            await self.adapter.close()

    @pytest.mark.asyncio
    async def test_chat_image_video_unsupported(self):
        """测试非语音能力抛出不支持异常"""
        from agn.core.errors import UnsupportedCapabilityError

        await self.adapter.start()
        try:
            # 测试普通异步方法
            for coro in [
                self.adapter.chat("sonic-2", []),
                self.adapter.image_generate("sonic-2", "test"),
                self.adapter.video_create("sonic-2", "test"),
                self.adapter.video_poll("tid"),
            ]:
                with pytest.raises(UnsupportedCapabilityError):
                    await coro

            # 测试异步生成器方法（chat_stream）
            with pytest.raises(UnsupportedCapabilityError):
                async for _ in self.adapter.chat_stream("sonic-2", []):
                    pass
        finally:
            await self.adapter.close()

    def test_error_responses(self):
        """测试 HTTP 错误处理"""
        from agn.core.errors import APIError, AuthenticationError, RateLimitError
        # 401
        resp = MagicMock(status_code=401)
        resp.json.return_value = {"message": "Unauthorized"}
        with pytest.raises(AuthenticationError):
            self.adapter._handle_error(resp)

        # 429
        resp.status_code = 429
        resp.json.return_value = {"message": "Rate limit"}
        with pytest.raises(RateLimitError):
            self.adapter._handle_error(resp)

        # 404 音色不存在
        resp.status_code = 404
        resp.json.return_value = {"message": "Voice not found"}
        with pytest.raises(APIError) as exc_info:
            self.adapter._handle_error(resp)
        assert exc_info.value.status_code == 404

        # 422 参数错误
        resp.status_code = 422
        resp.json.return_value = {"message": "Invalid output format"}
        with pytest.raises(APIError):
            self.adapter._handle_error(resp)


# ==================== AssemblyAI & Cartesia 路由和工厂注册测试 ====================


class TestRouterAssemblyAICartesiaMapping:
    """测试路由表包含 AssemblyAI 和 Cartesia 模型映射"""

    def test_router_has_assemblyai_models(self):
        from agn.router import Router
        MODEL_MAP = Router.MODEL_PROVIDER_MAP
        assert MODEL_MAP.get("best") == "assemblyai"
        assert MODEL_MAP.get("nano") == "assemblyai"

    def test_router_has_cartesia_models(self):
        from agn.router import Router
        MODEL_MAP = Router.MODEL_PROVIDER_MAP
        assert MODEL_MAP.get("sonic-2") == "cartesia"
        assert MODEL_MAP.get("sonic-turbo") == "cartesia"
        assert MODEL_MAP.get("sonic-2-2025-04-01") == "cartesia"
        assert MODEL_MAP.get("sonic-preview") == "cartesia"

    def test_adapter_factory_registration(self):
        from agn.adapters import AssemblyAIAdapter, CartesiaAdapter
        from agn.adapters.factory import AdapterFactory

        assert AdapterFactory.get_adapter_class("assemblyai") is AssemblyAIAdapter
        assert AdapterFactory.get_adapter_class("assembly") is AssemblyAIAdapter
        assert AdapterFactory.get_adapter_class("aai") is AssemblyAIAdapter

        assert AdapterFactory.get_adapter_class("cartesia") is CartesiaAdapter
        assert AdapterFactory.get_adapter_class("sonic") is CartesiaAdapter


# ==================== Edge TTS 免费语音合成单元测试 ====================


class TestEdgeTTSAdapter:
    """测试 Edge TTS 适配器"""

    def setup_method(self):
        self.config = ProviderConfig(provider_type="edge-tts", api_key="")
        self.adapter = EdgeTTSAdapter(config=self.config)

    def test_adapter_capabilities(self):
        """测试 Edge TTS 仅支持 TTS"""
        assert Capabilities.AUDIO_SPEECH in self.adapter.supported_capabilities
        assert Capabilities.AUDIO_TRANSCRIBE not in self.adapter.supported_capabilities
        assert self.adapter.provider_type == "edge-tts"
        assert self.adapter.provider_name == "Edge TTS"

    def test_adapter_requires_no_api_key(self):
        """测试 Edge TTS 标记为免费 Provider，无需 API Key"""
        assert self.adapter.requires_api_key is False

    def test_adapter_init_without_api_key(self):
        """测试 Edge TTS 不传 api_key 也能创建适配器"""
        config = ProviderConfig(provider_type="edge-tts")
        adapter = EdgeTTSAdapter(config=config)
        # api_key 为 None，适配器仍可正常创建
        assert adapter.config.api_key is None
        assert adapter.requires_api_key is False

    def test_default_voice_is_chinese(self):
        """测试默认音色是中文晓晓"""
        assert self.adapter.DEFAULT_VOICE == "zh-CN-XiaoxiaoNeural"

    def test_resolve_voice_chinese_short_names(self):
        """测试中文简称音色解析"""
        assert self.adapter._resolve_voice("晓晓") == "zh-CN-XiaoxiaoNeural"
        assert self.adapter._resolve_voice("xiaoxiao") == "zh-CN-XiaoxiaoNeural"
        assert self.adapter._resolve_voice("云希") == "zh-CN-YunxiNeural"
        assert self.adapter._resolve_voice("yunxi") == "zh-CN-YunxiNeural"

    def test_resolve_voice_english_short_names(self):
        """测试英文简称音色解析"""
        assert self.adapter._resolve_voice("jenny") == "en-US-JennyNeural"
        assert self.adapter._resolve_voice("nanami") == "ja-JP-NanamiNeural"

    def test_resolve_voice_full_id_passthrough(self):
        """测试完整 voice ID 原样返回"""
        full_id = "zh-CN-XiaoxiaoNeural"
        assert self.adapter._resolve_voice(full_id) == full_id
        full_id2 = "en-US-AriaNeural"
        assert self.adapter._resolve_voice(full_id2) == full_id2

    def test_resolve_voice_empty_uses_default(self):
        """测试空 voice 使用默认值"""
        assert self.adapter._resolve_voice("") == self.adapter.DEFAULT_VOICE

    def test_resolve_voice_case_insensitive(self):
        """测试大小写不敏感匹配"""
        assert self.adapter._resolve_voice("XIAOXIAO") == "zh-CN-XiaoxiaoNeural"
        assert self.adapter._resolve_voice("JENNY") == "en-US-JennyNeural"

    def test_resolve_voice_unknown_falls_back_to_default(self):
        """测试未知 voice 回退到默认"""
        assert self.adapter._resolve_voice("nonexistent-voice") == self.adapter.DEFAULT_VOICE

    def test_output_format_mp3(self):
        """测试 MP3 输出格式"""
        edge_fmt, content_type = self.adapter._get_output_format("mp3")
        assert "mp3" in edge_fmt
        assert content_type == "audio/mpeg"

    def test_output_format_wav(self):
        """测试 WAV 输出格式"""
        edge_fmt, content_type = self.adapter._get_output_format("wav")
        assert "pcm" in edge_fmt
        assert content_type == "audio/wav"

    def test_output_format_webm(self):
        """测试 WEBM 输出格式"""
        edge_fmt, content_type = self.adapter._get_output_format("webm")
        assert "opus" in edge_fmt
        assert content_type == "audio/ogg"

    def test_output_format_default_mp3(self):
        """测试默认输出格式是 MP3"""
        edge_fmt, content_type = self.adapter._get_output_format(None)
        assert "mp3" in edge_fmt
        assert content_type == "audio/mpeg"

    def test_output_format_high_quality_mp3(self):
        """测试高码率 MP3 格式"""
        edge_fmt, _ = self.adapter._get_output_format("mp3-128k")
        assert "128kbit" in edge_fmt

    def test_common_voices_has_chinese(self):
        """测试预设音色包含中文男女声"""
        voices = self.adapter.COMMON_VOICES
        zh_voices = [v for v in voices.values() if v.startswith("zh-CN")]
        assert len(zh_voices) >= 10

    def test_common_voices_has_english(self):
        """测试预设音色包含英文"""
        voices = self.adapter.COMMON_VOICES
        en_voices = [v for v in voices.values() if v.startswith("en-US")]
        assert len(en_voices) >= 5

    @pytest.mark.asyncio
    async def test_transcribe_raises_unsupported(self):
        """测试 ASR 能力不支持"""
        from agn.core.errors import UnsupportedCapabilityError
        self.adapter._edge_tts_module = MagicMock()
        with pytest.raises(UnsupportedCapabilityError):
            await self.adapter.transcribe("edge-tts", b"x")

    @pytest.mark.asyncio
    async def test_speech_with_mock_edge_tts(self):
        """测试 TTS 使用 mock 的 edge-tts 库"""
        mock_edge_tts = MagicMock()
        mock_communicate = MagicMock()

        async def fake_stream():
            yield {"type": "audio", "data": b"chunk1"}
            yield {"type": "WordBoundary", "offset": 100, "text": "hello"}
            yield {"type": "audio", "data": b"chunk2"}
            yield {"type": "audio", "data": b"chunk3"}

        mock_communicate.stream = fake_stream
        mock_edge_tts.Communicate = MagicMock(return_value=mock_communicate)
        self.adapter._edge_tts_module = mock_edge_tts

        result = await self.adapter.speech(
            "edge-tts",
            "Hello world",
            voice="晓晓",
        )

        assert isinstance(result, SpeechResult)
        assert result.audio_data == b"chunk1chunk2chunk3"
        assert result.format == "mp3"
        assert result.content_type == "audio/mpeg"
        assert result.model == "edge-tts"

        mock_edge_tts.Communicate.assert_called_once()
        call_kwargs = mock_edge_tts.Communicate.call_args.kwargs
        assert call_kwargs["text"] == "Hello world"
        assert call_kwargs["voice"] == "zh-CN-XiaoxiaoNeural"

    @pytest.mark.asyncio
    async def test_speech_with_rate_pitch_volume(self):
        """测试语速、音调、音量参数传递"""
        mock_edge_tts = MagicMock()
        mock_communicate = MagicMock()

        async def fake_stream():
            yield {"type": "audio", "data": b"audio"}

        mock_communicate.stream = fake_stream
        mock_edge_tts.Communicate = MagicMock(return_value=mock_communicate)
        self.adapter._edge_tts_module = mock_edge_tts

        await self.adapter.speech(
            "edge-tts", "test",
            voice="yunxi",
            rate="+20%",
            pitch="+50Hz",
            volume="-10%",
        )

        call_kwargs = mock_edge_tts.Communicate.call_args.kwargs
        assert call_kwargs["rate"] == "+20%"
        assert call_kwargs["pitch"] == "+50Hz"
        assert call_kwargs["volume"] == "-10%"
        assert call_kwargs["voice"] == "zh-CN-YunxiNeural"

    @pytest.mark.asyncio
    async def test_speech_wav_format(self):
        """测试 WAV 格式输出"""
        mock_edge_tts = MagicMock()
        mock_communicate = MagicMock()

        async def fake_stream():
            yield {"type": "audio", "data": b"wavdata"}

        mock_communicate.stream = fake_stream
        mock_edge_tts.Communicate = MagicMock(return_value=mock_communicate)
        self.adapter._edge_tts_module = mock_edge_tts

        result = await self.adapter.speech(
            "edge-tts", "test",
            output_format="wav",
        )
        assert result.format == "wav"
        assert result.content_type == "audio/wav"

    @pytest.mark.asyncio
    async def test_list_models(self):
        """测试列出 Edge TTS 模型"""
        self.adapter._edge_tts_module = MagicMock()
        models = await self.adapter.list_models()
        assert len(models) == 1
        m = models[0]
        assert m.id == "edge-tts"
        assert m.provider == "edge-tts"
        assert m.type == "audio"
        assert "audio_speech" in m.capabilities

    @pytest.mark.asyncio
    async def test_chat_image_video_unsupported(self):
        """测试非语音能力抛出不支持异常"""
        from agn.core.errors import UnsupportedCapabilityError

        self.adapter._edge_tts_module = MagicMock()

        # 测试普通异步方法
        for coro in [
            self.adapter.chat("edge-tts", []),
            self.adapter.image_generate("edge-tts", "test"),
            self.adapter.video_create("edge-tts", "test"),
            self.adapter.video_poll("tid"),
        ]:
            with pytest.raises(UnsupportedCapabilityError):
                await coro

        # 测试异步生成器方法（chat_stream）
        with pytest.raises(UnsupportedCapabilityError):
            async for _ in self.adapter.chat_stream("edge-tts", []):
                pass

    def test_get_edge_tts_raises_import_error_when_missing(self):
        """测试未安装 edge-tts 时抛出 ImportError"""
        adapter = EdgeTTSAdapter(config=self.config)
        adapter._edge_tts_module = None
        with patch.dict("sys.modules", {"edge_tts": None}):
            with pytest.raises(ImportError, match="edge-tts library not installed"):
                adapter._get_edge_tts()


class TestRouterEdgeTTSMapping:
    """测试路由表包含 Edge TTS 模型映射"""

    def test_router_has_edge_tts_models(self):
        from agn.router import Router
        MODEL_MAP = Router.MODEL_PROVIDER_MAP
        assert MODEL_MAP.get("edge-tts") == "edge-tts"
        assert MODEL_MAP.get("edge_tts") == "edge-tts"
        assert MODEL_MAP.get("edge-neural") == "edge-tts"

    def test_router_has_edge_tts_voice_shortcuts(self):
        """测试路由表包含常用语音快捷映射"""
        from agn.router import Router
        MODEL_MAP = Router.MODEL_PROVIDER_MAP
        assert MODEL_MAP.get("zh-CN-XiaoxiaoNeural") == "edge-tts"
        assert MODEL_MAP.get("zh-CN-YunxiNeural") == "edge-tts"
        assert MODEL_MAP.get("en-US-JennyNeural") == "edge-tts"

    def test_adapter_factory_registration(self):
        from agn.adapters import EdgeTTSAdapter
        from agn.adapters.factory import AdapterFactory

        assert AdapterFactory.get_adapter_class("edge-tts") is EdgeTTSAdapter
        assert AdapterFactory.get_adapter_class("edge_tts") is EdgeTTSAdapter
        assert AdapterFactory.get_adapter_class("edge") is EdgeTTSAdapter
        assert AdapterFactory.get_adapter_class("microsoft-tts") is EdgeTTSAdapter
