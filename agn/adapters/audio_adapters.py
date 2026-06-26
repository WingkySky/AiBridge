"""
AGN-SDK 语音专用适配器

实现主流语音服务提供商的 API 调用：
- ElevenLabs: 全球最流行的 TTS 服务，高质量多音色
- Deepgram: 全球最快的语音识别 ASR 服务，Nova 系列模型
- (后续可扩展: AssemblyAI, Play.ht, 百度语音 等)

API 文档:
- ElevenLabs: https://elevenlabs.io/docs/api-reference
- Deepgram: https://developers.deepgram.com/reference/listen-file
"""

import logging
from typing import Any

import httpx

from agn.adapters.base import BaseAdapter, Capabilities
from agn.adapters.factory import AdapterFactory
from agn.core.errors import (
    APIError,
    AuthenticationError,
    RateLimitError,
    UnsupportedCapabilityError,
)
from agn.models.audio import (
    SpeechResult,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)
from agn.models.chat import ChatMessage
from agn.models.common import ModelInfo, ProviderConfig

logger = logging.getLogger(__name__)


# ==================== ElevenLabs TTS 适配器 ====================


class ElevenLabsAdapter(BaseAdapter):
    """
    ElevenLabs 适配器

    全球最流行的高质量语音合成平台。

    API 规范：
    - Base URL: https://api.elevenlabs.io/v1
    - TTS: POST /text-to-speech/{voice_id}
    - TTS Streaming: POST /text-to-speech/{voice_id}/stream
    - 认证: xi-api-key header
    - 文档: https://elevenlabs.io/docs/api-reference
    - 特点: 超逼真音色、多语言支持、声音克隆
    """

    provider_type = "elevenlabs"
    provider_name = "ElevenLabs"
    supported_capabilities = [
        Capabilities.AUDIO_SPEECH,
    ]

    DEFAULT_BASE_URL = "https://api.elevenlabs.io/v1"

    # ElevenLabs 内置音色 ID
    DEFAULT_VOICES = {
        "Rachel": "21m00Tcm4TlvDq8ikWAM",
        "Drew": "29vD33N1CtxCmqQRPOHJ",
        "Clyde": "2EiwWnXFnvU5JabPnv8n",
        "Paul": "5Q0t7uMcjvnagumLfvZi",
        "Domi": "AZnzlk1XvdvUeBnXmlld",
        "Dave": "CYw3kZ02Hs0563khs1Fj",
        "Fin": "D38z5RcWu1voky8WS1ja",
        "Sarah": "EXAVITQu4vr4xnSDxMaL",
        "Antoni": "ErXwobaYiN019PkySvjV",
        "Thomas": "GBv7mTt0atIp3Br8iCZE",
        "Charlie": "IKne3meq5aSn9XLyUdCD",
        "George": "JBFqnCBsd32t6Ie9FZ2Q",
        "Emily": "LcfcDJNUP1GQjkzn1xUU",
        "Elli": "MF3mGyEYCl7XYWbV9V6O",
        "Callum": "N2lVS1w4EtoT3dr4eOWO",
        "Patrick": "ODq5zmih8GrVes37DekR",
        "Harry": "SOYHLrjzK2X1ezoPC6cr",
        "Liam": "TX3LPaxmHKxFdv7VOQHJ",
        "Dorothy": "ThT5KcBeYPX3keUQqHPh",
        "Josh": "TxGEqnHWrfWFTfGW9XjX",
        "Arnold": "VR6AewLTigWG4xSOukaG",
        "Charlotte": "XB0fDUnXU5powFXDhCwa",
        "Alice": "Xb7hH8MSUJpSbSDYk0k2",
        "Matilda": "XrExE9yKIg1WjnnlVkGX",
        "Matthew": "Yko7PKHZNXotIFUBG7I9",
        "James": "ZQe5CZNOzWyzPSCn5a3c",
        "Joseph": "Zlb1dXrM653N07WRPnSh",
        "Jeremy": "bVMeCyTHy58xNoL34h3p",
        "Michael": "flq6f7yk4E4fJM5XTYuZ",
        "Ethan": "g5CIjZEefAph4nQFvHAz",
        "Chris": "iP95p4xoKVk53GoZ742B",
        "Gigi": "jBpfuIE2acCO8z3wKNLl",
        "Freya": "jsCqWAovK2LkecY7zXl4",
        "Brian": "nPczCjzI2devxB14oouP",
        "Grace": "oWAxZDx7w5VEj9dCyTzz",
        "Daniel": "onwK4e9ZLuTAKqWW03F9",
        "Lily": "pFZP5JQG7iQjIQuC4Bku",
        "Serena": "pMsXgVXv3BLzUgSXRplE",
        "Adam": "pNInz6obpgDQGcFmaJgB",
        "Nicole": "piTKgcLEGmPE4e6mEKli",
        "Bill": "pqHfZKP75CvOlQylNhV4",
        "Jessie": "t0jbNlBVZ17f02VDIeMI",
        "Sam": "yoZ06aMxZJJ28mfd3POQ",
        "Glinda": "z9fAnlkpzviVjnFOo0Tc",
        "Giovanni": "zcAOhNBS3c14rBihAFp1",
        "Mimi": "zrHiDhphv9ZnVXBqCLjz",
    }

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self.api_key = config.api_key or ""
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """启动适配器，初始化 HTTP 客户端"""
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
        )

    async def close(self) -> None:
        """关闭适配器，释放 HTTP 客户端"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端，未启动则抛出异常"""
        if self._http_client is None:
            raise RuntimeError("Adapter not started. Call start() first.")
        return self._http_client

    def _get_voice_id(self, voice: str) -> str:
        """
        根据 voice 名称或 ID 获取 ElevenLabs voice_id

        Args:
            voice: 音色名称（如 "Rachel"）或直接是 voice_id

        Returns:
            ElevenLabs voice_id
        """
        if voice in self.DEFAULT_VOICES:
            return self.DEFAULT_VOICES[voice]
        return voice

    async def speech(
        self,
        model: str,
        input: str,
        voice: str = "Rachel",
        **kwargs: Any,
    ) -> SpeechResult:
        """
        文字转语音 (ElevenLabs TTS)

        Args:
            model: 模型 ID (eleven_monolingual_v1 / eleven_multilingual_v1 / eleven_multilingual_v2 / eleven_turbo_v2)
            input: 要合成的文本
            voice: 音色名称或 voice_id
            **kwargs: 其他参数
                - stability: float 稳定性 (0-1)
                - similarity_boost: float 相似度增强 (0-1)
                - style: float 风格 (0-1，仅 v2 模型)
                - use_speaker_boost: bool 说话人增强
                - response_format: str 输出格式 (mp3/wav/ogg/ulaw/pcm)
                - output_format: str 同 response_format

        Returns:
            SpeechResult 包含音频二进制数据
        """
        client = self._get_client()
        voice_id = self._get_voice_id(voice)

        response_format = kwargs.get("response_format") or kwargs.get(
            "output_format", "mp3"
        )

        payload: dict[str, Any] = {
            "text": input,
            "model_id": model,
        }

        voice_settings: dict[str, Any] = {}
        if "stability" in kwargs:
            voice_settings["stability"] = kwargs["stability"]
        if "similarity_boost" in kwargs:
            voice_settings["similarity_boost"] = kwargs["similarity_boost"]
        if "style" in kwargs:
            voice_settings["style"] = kwargs["style"]
        if "use_speaker_boost" in kwargs:
            voice_settings["use_speaker_boost"] = kwargs["use_speaker_boost"]

        if voice_settings:
            payload["voice_settings"] = voice_settings

        query_params = {}
        if response_format != "mp3":
            query_params["output_format"] = response_format

        content_type_map = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "opus": "audio/opus",
            "ulaw": "audio/basic",
            "pcm": "audio/pcm",
        }

        headers = {"Accept": content_type_map.get(response_format, "audio/mpeg")}

        response = await client.post(
            f"/text-to-speech/{voice_id}",
            json=payload,
            params=query_params,
            headers=headers,
        )
        self._handle_error(response)

        return SpeechResult(
            audio_data=response.content,
            content_type=content_type_map.get(response_format, "audio/mpeg"),
            format=response_format,
            model=model,
        )

    async def transcribe(
        self,
        model: str,
        file: Any,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """ElevenLabs 目前不提供语音转文字服务"""
        raise UnsupportedCapabilityError(
            message="ElevenLabs does not support speech-to-text (transcription)",
            details={"provider": self.provider_type, "capability": "audio_transcribe"},
        )

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="ElevenLabs does not support chat completions",
            details={"provider": self.provider_type, "capability": "chat"},
        )

    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        if False:
            yield
        raise UnsupportedCapabilityError(
            message="ElevenLabs does not support streaming chat",
            details={"provider": self.provider_type, "capability": "chat_stream"},
        )

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="ElevenLabs does not support image generation",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="ElevenLabs does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="ElevenLabs does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """列出 ElevenLabs 可用模型"""
        models = [
            ModelInfo(
                id="eleven_multilingual_v2",
                name="Eleven Multilingual v2",
                type="audio",
                provider="elevenlabs",
                capabilities=["audio_speech"],
                description="多语言v2模型，支持29种语言，最新最稳定",
            ),
            ModelInfo(
                id="eleven_multilingual_v1",
                name="Eleven Multilingual v1",
                type="audio",
                provider="elevenlabs",
                capabilities=["audio_speech"],
                description="多语言v1模型",
            ),
            ModelInfo(
                id="eleven_monolingual_v1",
                name="Eleven English v1",
                type="audio",
                provider="elevenlabs",
                capabilities=["audio_speech"],
                description="英语单语言模型",
            ),
            ModelInfo(
                id="eleven_turbo_v2_5",
                name="Eleven Turbo v2.5",
                type="audio",
                provider="elevenlabs",
                capabilities=["audio_speech"],
                description="Turbo v2.5 极速模型，低延迟",
            ),
            ModelInfo(
                id="eleven_turbo_v2",
                name="Eleven Turbo v2",
                type="audio",
                provider="elevenlabs",
                capabilities=["audio_speech"],
                description="Turbo v2 极速模型",
            ),
            ModelInfo(
                id="eleven_flash_v2_5",
                name="Eleven Flash v2.5",
                type="audio",
                provider="elevenlabs",
                capabilities=["audio_speech"],
                description="Flash v2.5 最快模型，超低延迟",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    def _handle_error(self, response: httpx.Response) -> None:
        """处理 ElevenLabs API 错误响应"""
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid ElevenLabs API key (xi-api-key)")
        if response.status_code == 429:
            raise RateLimitError(
                message="ElevenLabs rate limit exceeded or quota exhausted"
            )
        if response.status_code == 404:
            raise APIError(
                message="ElevenLabs voice_id or model not found", status_code=404
            )
        if response.status_code in (400, 422):
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", {})
                if isinstance(error_detail, dict):
                    error_msg = error_detail.get("message", str(error_detail))
                elif isinstance(error_detail, list):
                    error_msg = "; ".join(
                        str(e.get("msg", e))
                        for e in error_detail
                        if isinstance(e, dict)
                    )
                else:
                    error_msg = str(error_detail)
            except Exception:
                error_msg = f"Bad request: {response.text[:200]}"
            raise APIError(message=error_msg, status_code=response.status_code)

        try:
            error_data = response.json()
            error_msg = (
                error_data.get("detail", {}).get("message")
                if isinstance(error_data.get("detail"), dict)
                else error_data.get(
                    "message", error_data.get("detail", f"HTTP {response.status_code}")
                )
            )
        except Exception:
            error_msg = f"HTTP {response.status_code}"

        raise APIError(message=str(error_msg), status_code=response.status_code)


# ==================== Deepgram ASR 适配器 ====================


class DeepgramAdapter(BaseAdapter):
    """
    Deepgram 适配器

    全球最快的语音识别(ASR)服务之一，Nova 系列模型，以超低延迟和高准确率著称。

    API 规范：
    - Base URL: https://api.deepgram.com/v1
    - 预录音频转写: POST /listen
    - 认证: Authorization: Token <API_KEY>
    - 文档: https://developers.deepgram.com/reference/listen-file
    - 特点: 超低延迟、Nova-2 模型准确率高、支持说话人分离、关键词搜索、智能格式化
    """

    provider_type = "deepgram"
    provider_name = "Deepgram"
    supported_capabilities = [
        Capabilities.AUDIO_TRANSCRIBE,
    ]

    DEFAULT_BASE_URL = "https://api.deepgram.com/v1"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self.api_key = config.api_key or ""
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """启动适配器，初始化 HTTP 客户端"""
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "Authorization": f"Token {self.api_key}",
            },
        )

    async def close(self) -> None:
        """关闭适配器，释放 HTTP 客户端"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端，未启动则抛出异常"""
        if self._http_client is None:
            raise RuntimeError("Adapter not started. Call start() first.")
        return self._http_client

    async def get_audio_bytes(self, file: Any) -> tuple[bytes, str]:
        """
        统一音频输入处理，支持 bytes / 文件路径 / URL / base64 / 文件对象

        Args:
            file: 音频输入（bytes/str/Path/file-like/base64）

        Returns:
            (audio_bytes, mime_type)
        """
        import base64
        from pathlib import Path

        if isinstance(file, bytes):
            return file, "audio/wav"

        if isinstance(file, (str, Path)):
            file_str = str(file)
            if file_str.startswith(("http://", "https://")):
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self.config.timeout)
                ) as http:
                    resp = await http.get(file_str)
                    resp.raise_for_status()
                    return resp.content, "audio/mpeg"
            if file_str.startswith("data:"):
                header, encoded = file_str.split(",", 1)
                mime = "audio/wav"
                for fmt in ["wav", "mp3", "ogg", "flac", "webm", "m4a"]:
                    if fmt in header:
                        mime = f"audio/{fmt}" if fmt != "mp3" else "audio/mpeg"
                        break
                return base64.b64decode(encoded), mime
            path = Path(file_str)
            if path.exists():
                ext = path.suffix.lower().lstrip(".")
                mime_map = {
                    "mp3": "audio/mpeg",
                    "wav": "audio/wav",
                    "ogg": "audio/ogg",
                    "flac": "audio/flac",
                    "m4a": "audio/mp4",
                    "webm": "audio/webm",
                    "mp4": "audio/mp4",
                    "aac": "audio/aac",
                    "opus": "audio/opus",
                }
                return path.read_bytes(), mime_map.get(ext, "audio/wav")
            # 判断是否像文件路径（包含路径分隔符或常见音频扩展名）
            audio_exts = {
                ".wav",
                ".mp3",
                ".ogg",
                ".flac",
                ".m4a",
                ".webm",
                ".mp4",
                ".aac",
                ".opus",
                ".pcm",
                ".ulaw",
            }
            looks_like_path = (
                "/" in file_str
                or "\\" in file_str
                or path.suffix.lower() in audio_exts
                or file_str.startswith(("~", "./", "../", "/"))
            )
            if looks_like_path:
                raise FileNotFoundError(f"Audio file not found: {file_str}")
            try:
                decoded = base64.b64decode(file_str, validate=True)
                if len(decoded) > 10:
                    return decoded, "audio/wav"
            except Exception:
                pass

        if hasattr(file, "read"):
            data = file.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            name = getattr(file, "name", "audio.wav")
            ext = Path(str(name)).suffix.lower().lstrip(".")
            mime_map = {
                "mp3": "audio/mpeg",
                "wav": "audio/wav",
                "ogg": "audio/ogg",
                "flac": "audio/flac",
                "m4a": "audio/mp4",
                "webm": "audio/webm",
            }
            return data, mime_map.get(ext, "audio/wav")

        raise ValueError(f"Unsupported file input type: {type(file)}")

    async def transcribe(
        self,
        model: str,
        file: Any,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        语音转文字 (Deepgram ASR)

        Args:
            model: 模型 ID (nova-2 / nova-2-meeting / nova-2-phonecall / whisper-large 等)
            file: 音频文件（bytes/路径/URL/base64/文件对象）
            **kwargs: Deepgram 特定参数
                - language: str 语言代码 (zh/en/ja 等，默认自动检测)
                - smart_format: bool 智能格式化（标点、数字等，默认 True）
                - diarize: bool 说话人分离（默认 False）
                - punctuate: bool 标点符号（默认 True）
                - profanity_filter: bool 脏话过滤（默认 False）
                - keywords: list[str] 关键词提示
                - utterances: bool 按语句分割
                - replace: list[dict] 自定义替换规则
                - tag: list[str] 自定义标签

        Returns:
            TranscriptionResult 统一转写结果
        """
        client = self._get_client()

        audio_data, content_type = await self.get_audio_bytes(file)

        query_params: dict[str, Any] = {
            "model": model,
        }

        smart_format = kwargs.get("smart_format", True)
        if smart_format:
            query_params["smart_format"] = "true"

        punctuate = kwargs.get("punctuate", True)
        if punctuate:
            query_params["punctuate"] = "true"

        language = kwargs.get("language")
        if language:
            query_params["language"] = language

        diarize = kwargs.get("diarize", False)
        if diarize:
            query_params["diarize"] = "true"

        profanity_filter = kwargs.get("profanity_filter", False)
        if profanity_filter:
            query_params["profanity_filter"] = "true"

        utterances = kwargs.get("utterances", False)
        if utterances:
            query_params["utterances"] = "true"

        keywords = kwargs.get("keywords")
        if keywords and isinstance(keywords, list):
            query_params["keywords"] = keywords

        response = await client.post(
            "/listen",
            content=audio_data,
            params=query_params,
            headers={"Content-Type": content_type},
        )
        self._handle_error(response)
        result = response.json()

        return self._parse_deepgram_response(result, model=model)

    def _parse_deepgram_response(
        self,
        result: dict[str, Any],
        model: str,
    ) -> TranscriptionResult:
        """
        解析 Deepgram 响应为统一 TranscriptionResult 格式

        Deepgram 响应结构：
        {
          "results": {
            "channels": [{
              "alternatives": [{
                "transcript": "...",
                "confidence": 0.99,
                "words": [{...}],
                "paragraphs": {...}
              }]
            }]
          },
          "metadata": {
            "duration": 3.5,
            "model_info": {"name": "nova-2-general"}
          }
        }
        """
        results = result.get("results", {})
        channels = results.get("channels", [])

        full_text_parts: list[str] = []
        all_words: list[TranscriptionWord] = []
        all_segments: list[TranscriptionSegment] = []

        language = None
        duration = None

        metadata = result.get("metadata", {})
        if metadata:
            duration = metadata.get("duration")
            model_info = metadata.get("model_info", {})
            if isinstance(model_info, dict):
                language = model_info.get("language")

        for channel_idx, channel in enumerate(channels):
            alternatives = channel.get("alternatives", [])
            if not alternatives:
                continue

            alt = alternatives[0]
            transcript = alt.get("transcript", "")
            if transcript:
                full_text_parts.append(transcript)

            words = alt.get("words", [])
            for w in words:
                word = TranscriptionWord(
                    word=w.get("word", ""),
                    start=w.get("start", 0),
                    end=w.get("end", 0),
                )
                confidence = w.get("confidence")
                if confidence is not None:
                    word.confidence = confidence
                speaker = w.get("speaker")
                if speaker is not None:
                    word.speaker = speaker
                all_words.append(word)

            paragraphs = alt.get("paragraphs", {})
            if paragraphs and "paragraphs" in paragraphs:
                for para in paragraphs["paragraphs"]:
                    sentences = para.get("sentences", [])
                    for _sent_idx, sent in enumerate(sentences):
                        seg = TranscriptionSegment(
                            id=len(all_segments),
                            start=sent.get("start", 0),
                            end=sent.get("end", 0),
                            text=sent.get("text", ""),
                        )
                        if channel_idx > 0:
                            seg.channel = channel_idx
                        speaker = para.get("speaker")
                        if speaker is not None:
                            seg.speaker = speaker
                        all_segments.append(seg)

        full_text = " ".join(full_text_parts).strip()

        return TranscriptionResult(
            text=full_text,
            language=language,
            duration=duration,
            segments=all_segments if all_segments else None,
            words=all_words if all_words else None,
            task="transcribe",
            model=model,
        )

    async def speech(
        self,
        model: str,
        input: str,
        voice: str = "",
        **kwargs: Any,
    ) -> SpeechResult:
        """Deepgram Aura TTS（如果需要可以后续扩展），当前版本仅支持 ASR"""
        raise UnsupportedCapabilityError(
            message="Deepgram speech (TTS) is not implemented in this version; use ElevenLabs/OpenAI TTS instead",
            details={"provider": self.provider_type, "capability": "audio_speech"},
        )

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Deepgram does not support chat completions",
            details={"provider": self.provider_type, "capability": "chat"},
        )

    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        if False:
            yield
        raise UnsupportedCapabilityError(
            message="Deepgram does not support streaming chat",
            details={"provider": self.provider_type, "capability": "chat_stream"},
        )

    async def image_generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Deepgram does not support image generation",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Deepgram does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(
        self,
        task_id: str,
        model: str = "",
    ) -> Any:
        raise UnsupportedCapabilityError(
            message="Deepgram does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(
        self,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        """列出 Deepgram 可用 ASR 模型"""
        models = [
            ModelInfo(
                id="nova-3",
                name="Nova 3",
                type="audio",
                provider="deepgram",
                capabilities=["audio_transcribe"],
                description="Nova 3 最新通用模型，最高准确率",
            ),
            ModelInfo(
                id="nova-2",
                name="Nova 2",
                type="audio",
                provider="deepgram",
                capabilities=["audio_transcribe"],
                description="Nova 2 通用模型，推荐默认使用",
            ),
            ModelInfo(
                id="nova-2-general",
                name="Nova 2 General",
                type="audio",
                provider="deepgram",
                capabilities=["audio_transcribe"],
                description="Nova 2 通用场景",
            ),
            ModelInfo(
                id="nova-2-meeting",
                name="Nova 2 Meeting",
                type="audio",
                provider="deepgram",
                capabilities=["audio_transcribe"],
                description="Nova 2 会议场景优化（多人对话）",
            ),
            ModelInfo(
                id="nova-2-phonecall",
                name="Nova 2 Phone Call",
                type="audio",
                provider="deepgram",
                capabilities=["audio_transcribe"],
                description="Nova 2 电话通话优化（8kHz音频）",
            ),
            ModelInfo(
                id="nova-2-conversationalai",
                name="Nova 2 Conversational AI",
                type="audio",
                provider="deepgram",
                capabilities=["audio_transcribe"],
                description="Nova 2 对话AI/语音助手优化",
            ),
            ModelInfo(
                id="nova-2-video",
                name="Nova 2 Video",
                type="audio",
                provider="deepgram",
                capabilities=["audio_transcribe"],
                description="Nova 2 视频/播客/多说话人场景",
            ),
            ModelInfo(
                id="nova-2-medical",
                name="Nova 2 Medical",
                type="audio",
                provider="deepgram",
                capabilities=["audio_transcribe"],
                description="Nova 2 医疗领域优化",
            ),
            ModelInfo(
                id="nova-2-finance",
                name="Nova 2 Finance",
                type="audio",
                provider="deepgram",
                capabilities=["audio_transcribe"],
                description="Nova 2 金融领域优化",
            ),
            ModelInfo(
                id="nova-2-drivethru",
                name="Nova 2 Drive-Thru",
                type="audio",
                provider="deepgram",
                capabilities=["audio_transcribe"],
                description="Nova 2 餐厅免下车窗口优化",
            ),
            ModelInfo(
                id="whisper-large",
                name="Whisper Large (Deepgram)",
                type="audio",
                provider="deepgram",
                capabilities=["audio_transcribe"],
                description="OpenAI Whisper Large 托管版",
            ),
            ModelInfo(
                id="whisper-medium",
                name="Whisper Medium (Deepgram)",
                type="audio",
                provider="deepgram",
                capabilities=["audio_transcribe"],
                description="OpenAI Whisper Medium 托管版",
            ),
            ModelInfo(
                id="whisper-small",
                name="Whisper Small (Deepgram)",
                type="audio",
                provider="deepgram",
                capabilities=["audio_transcribe"],
                description="OpenAI Whisper Small 托管版",
            ),
            ModelInfo(
                id="enhanced",
                name="Enhanced",
                type="audio",
                provider="deepgram",
                capabilities=["audio_transcribe"],
                description="Enhanced 增强模型（旧版，兼容使用）",
            ),
            ModelInfo(
                id="base",
                name="Base",
                type="audio",
                provider="deepgram",
                capabilities=["audio_transcribe"],
                description="Base 基础模型（最快、成本最低）",
            ),
        ]

        if model_type:
            models = [m for m in models if m.type == model_type]

        return models

    def _handle_error(self, response: httpx.Response) -> None:
        """处理 Deepgram API 错误响应"""
        if response.status_code < 400:
            return

        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Deepgram API key (Token)")
        if response.status_code == 403:
            raise AuthenticationError(
                message="Deepgram API key lacks permission or account suspended"
            )
        if response.status_code == 429:
            raise RateLimitError(
                message="Deepgram rate limit exceeded or quota exhausted"
            )
        if response.status_code in (400, 422):
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    err_msg = (
                        error_data.get("err_msg")
                        or error_data.get("message")
                        or error_data.get("error", "")
                    )
                    if isinstance(err_msg, dict):
                        err_msg = err_msg.get("message", str(err_msg))
                    if not err_msg:
                        err_msg = str(error_data)
                else:
                    err_msg = str(error_data)
            except Exception:
                err_msg = f"Bad request: {response.text[:200]}"
            raise APIError(message=str(err_msg), status_code=response.status_code)

        try:
            error_data = response.json()
            if isinstance(error_data, dict):
                err_msg = (
                    error_data.get("err_msg")
                    or error_data.get("message")
                    or error_data.get("error", f"HTTP {response.status_code}")
                )
                if isinstance(err_msg, dict):
                    err_msg = err_msg.get("message", str(err_msg))
            else:
                err_msg = f"HTTP {response.status_code}"
        except Exception:
            err_msg = f"HTTP {response.status_code}"

        raise APIError(message=str(err_msg), status_code=response.status_code)


# ==================== AssemblyAI 企业级 ASR 适配器 ====================


class AssemblyAIAdapter(BaseAdapter):
    """
    AssemblyAI 适配器

    企业级语音识别服务，支持丰富的语音理解能力。

    API 规范：
    - Base URL: https://api.assemblyai.com/v2
    - 上传音频: POST /upload (返回 upload_url)
    - 提交转写: POST /transcript (异步任务)
    - 查询结果: GET /transcript/{id} (轮询直到完成)
    - 认证: Authorization: <API_KEY> header
    - 文档: https://www.assemblyai.com/docs
    - 特点: 企业级ASR、说话人分离、情感分析、PII脱敏、章节检测、实体识别
    """

    provider_type = "assemblyai"
    provider_name = "AssemblyAI"
    supported_capabilities = [
        Capabilities.AUDIO_TRANSCRIBE,
    ]

    DEFAULT_BASE_URL = "https://api.assemblyai.com/v2"
    POLL_INTERVAL = 1.0
    MAX_POLLS = 300

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self.api_key = config.api_key or ""
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """启动适配器，初始化 HTTP 客户端"""
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "Authorization": self.api_key,
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        """关闭适配器，释放 HTTP 客户端"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端，未启动则抛出异常"""
        if self._http_client is None:
            raise RuntimeError("Adapter not started. Call start() first.")
        return self._http_client

    async def get_audio_bytes(self, file: Any) -> tuple[bytes, str]:
        """
        统一音频输入处理（复用 Deepgram 的实现模式）

        Args:
            file: 音频输入

        Returns:
            (audio_bytes, mime_type)
        """
        import base64
        from pathlib import Path

        if isinstance(file, bytes):
            return file, "application/octet-stream"

        if isinstance(file, (str, Path)):
            file_str = str(file)
            if file_str.startswith(("http://", "https://")):
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self.config.timeout)
                ) as http:
                    resp = await http.get(file_str)
                    resp.raise_for_status()
                    return resp.content, "application/octet-stream"
            if file_str.startswith("data:"):
                header, encoded = file_str.split(",", 1)
                return base64.b64decode(encoded), "application/octet-stream"
            path = Path(file_str)
            if path.exists():
                return path.read_bytes(), "application/octet-stream"
            audio_exts = {
                ".wav",
                ".mp3",
                ".ogg",
                ".flac",
                ".m4a",
                ".webm",
                ".mp4",
                ".aac",
                ".opus",
            }
            looks_like_path = (
                "/" in file_str
                or "\\" in file_str
                or path.suffix.lower() in audio_exts
                or file_str.startswith(("~", "./", "../", "/"))
            )
            if looks_like_path:
                raise FileNotFoundError(f"Audio file not found: {file_str}")
            try:
                decoded = base64.b64decode(file_str, validate=True)
                if len(decoded) > 10:
                    return decoded, "application/octet-stream"
            except Exception:
                pass

        if hasattr(file, "read"):
            data = file.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            return data, "application/octet-stream"

        raise ValueError(f"Unsupported file input type: {type(file)}")

    async def _upload_audio(self, audio_data: bytes) -> str:
        """
        上传音频到 AssemblyAI，返回 upload_url

        Args:
            audio_data: 音频二进制数据

        Returns:
            AssemblyAI upload_url
        """
        client = self._get_client()
        response = await client.post(
            "/upload",
            content=audio_data,
            headers={"Content-Type": "application/octet-stream"},
        )
        self._handle_error(response)
        result = response.json()
        return str(result["upload_url"])

    async def transcribe(
        self,
        model: str,
        file: Any,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        语音转文字 (AssemblyAI ASR)

        AssemblyAI 使用异步任务模式：上传 -> 提交 -> 轮询结果

        Args:
            model: 模型 ID (best/nano，默认为 best)
            file: 音频文件
            **kwargs: AssemblyAI 特定参数
                - language_code: str 语言代码 (en/zh/ja/es/fr 等)
                - speaker_labels: bool 说话人分离
                - punctuate: bool 标点（默认 True）
                - format_text: bool 格式化文本（默认 True）
                - filter_profanity: bool 脏话过滤
                - sentiment_analysis: bool 情感分析
                - auto_chapters: bool 自动章节
                - entity_detection: bool 实体识别
                - redact_pii: bool PII脱敏
                - audio_url: str 直接提供AssemblyAI可访问的URL（跳过上传）
                - polling_interval: float 轮询间隔秒（默认1.0）
                - max_polls: int 最大轮询次数（默认300）

        Returns:
            TranscriptionResult
        """
        import asyncio

        client = self._get_client()

        direct_url = kwargs.get("audio_url")
        if direct_url:
            audio_url = direct_url
        else:
            audio_data, _ = await self.get_audio_bytes(file)
            audio_url = await self._upload_audio(audio_data)

        speech_model = model if model in ("best", "nano") else "best"

        transcript_request: dict[str, Any] = {
            "audio_url": audio_url,
            "speech_model": speech_model,
            "punctuate": kwargs.get("punctuate", True),
            "format_text": kwargs.get("format_text", True),
        }

        if "language_code" in kwargs:
            transcript_request["language_code"] = kwargs["language_code"]
        if kwargs.get("speaker_labels"):
            transcript_request["speaker_labels"] = True
        if kwargs.get("filter_profanity"):
            transcript_request["filter_profanity"] = True
        if kwargs.get("sentiment_analysis"):
            transcript_request["sentiment_analysis"] = True
        if kwargs.get("auto_chapters"):
            transcript_request["auto_chapters"] = True
        if kwargs.get("entity_detection"):
            transcript_request["entity_detection"] = True
        if kwargs.get("redact_pii"):
            transcript_request["redact_pii_policies"] = kwargs.get(
                "redact_pii_policies", []
            )
            transcript_request["redact_pii"] = True
        if "word_boost" in kwargs and isinstance(kwargs["word_boost"], list):
            transcript_request["word_boost"] = kwargs["word_boost"]

        response = await client.post("/transcript", json=transcript_request)
        self._handle_error(response)
        submit_result = response.json()
        transcript_id = submit_result["id"]

        polling_interval = kwargs.get("polling_interval", self.POLL_INTERVAL)
        max_polls = kwargs.get("max_polls", self.MAX_POLLS)

        result_data = None
        for _ in range(max_polls):
            poll_resp = await client.get(f"/transcript/{transcript_id}")
            self._handle_error(poll_resp)
            poll_data = poll_resp.json()
            status = poll_data.get("status", "")
            if status == "completed":
                result_data = poll_data
                break
            if status == "error":
                error_msg = poll_data.get("error", "Transcription failed")
                raise APIError(
                    message=f"AssemblyAI transcription error: {error_msg}",
                    status_code=500,
                )
            await asyncio.sleep(polling_interval)

        if result_data is None:
            raise APIError(
                message="AssemblyAI transcription timed out", status_code=408
            )

        return self._parse_assemblyai_response(result_data, model=speech_model)

    def _parse_assemblyai_response(
        self,
        result: dict[str, Any],
        model: str,
    ) -> TranscriptionResult:
        """解析 AssemblyAI 响应为统一格式"""
        from agn.models.audio import TranscriptionSegment, TranscriptionWord

        full_text = result.get("text", "")

        words: list[TranscriptionWord] = []
        segments: list[TranscriptionSegment] = []
        language = result.get("language_code")
        duration = result.get("audio_duration")

        utterances = result.get("utterances") or []
        if utterances:
            for utt_idx, utt in enumerate(utterances):
                seg = TranscriptionSegment(
                    id=utt_idx,
                    start=utt.get("start", 0) / 1000.0,
                    end=utt.get("end", 0) / 1000.0,
                    text=utt.get("text", ""),
                    confidence=utt.get("confidence"),
                    speaker=(
                        str(utt.get("speaker"))
                        if utt.get("speaker") is not None
                        else None
                    ),
                )
                segments.append(seg)
                for w in utt.get("words", []):
                    words.append(
                        TranscriptionWord(
                            word=w.get("text", ""),
                            start=w.get("start", 0) / 1000.0,
                            end=w.get("end", 0) / 1000.0,
                            confidence=w.get("confidence"),
                            speaker=(
                                str(w.get("speaker"))
                                if w.get("speaker") is not None
                                else None
                            ),
                        )
                    )
        else:
            api_words = result.get("words") or []
            for w in api_words:
                words.append(
                    TranscriptionWord(
                        word=w.get("text", ""),
                        start=w.get("start", 0) / 1000.0,
                        end=w.get("end", 0) / 1000.0,
                        confidence=w.get("confidence"),
                    )
                )

        return TranscriptionResult(
            text=full_text,
            language=language,
            duration=duration,
            segments=segments if segments else None,
            words=words if words else None,
            task="transcribe",
            model=model,
        )

    async def speech(
        self,
        model: str,
        input: str,
        voice: str = "",
        **kwargs: Any,
    ) -> SpeechResult:
        """AssemblyAI 专注 ASR，不提供 TTS"""
        raise UnsupportedCapabilityError(
            message="AssemblyAI does not support text-to-speech",
            details={"provider": self.provider_type, "capability": "audio_speech"},
        )

    async def chat(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> Any:
        raise UnsupportedCapabilityError(
            message="AssemblyAI does not support chat completions",
            details={"provider": self.provider_type, "capability": "chat"},
        )

    async def chat_stream(
        self, model: str, messages: list[ChatMessage], **kwargs: Any
    ) -> Any:
        if False:
            yield
        raise UnsupportedCapabilityError(
            message="AssemblyAI does not support streaming chat",
            details={"provider": self.provider_type, "capability": "chat_stream"},
        )

    async def image_generate(self, model: str, prompt: str, **kwargs: Any) -> Any:
        raise UnsupportedCapabilityError(
            message="AssemblyAI does not support image generation",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(self, model: str, prompt: str, **kwargs: Any) -> Any:
        raise UnsupportedCapabilityError(
            message="AssemblyAI does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(self, task_id: str, model: str = "") -> Any:
        raise UnsupportedCapabilityError(
            message="AssemblyAI does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(self, model_type: str | None = None) -> list[ModelInfo]:
        """列出 AssemblyAI 可用模型"""
        models = [
            ModelInfo(
                id="best",
                name="Best",
                type="audio",
                provider="assemblyai",
                capabilities=["audio_transcribe"],
                description="最高准确率模型（默认），支持所有高级功能",
            ),
            ModelInfo(
                id="nano",
                name="Nano",
                type="audio",
                provider="assemblyai",
                capabilities=["audio_transcribe"],
                description="轻量模型，更低成本，适合简单场景",
            ),
        ]
        if model_type:
            models = [m for m in models if m.type == model_type]
        return models

    def _handle_error(self, response: httpx.Response) -> None:
        """处理 AssemblyAI 错误响应"""
        if response.status_code < 400:
            return
        if response.status_code == 401:
            raise AuthenticationError(message="Invalid AssemblyAI API key")
        if response.status_code == 429:
            raise RateLimitError(
                message="AssemblyAI rate limit exceeded or quota exhausted"
            )
        try:
            error_data = response.json()
            err_msg = error_data.get("error") or error_data.get(
                "message", f"HTTP {response.status_code}"
            )
        except Exception:
            err_msg = f"HTTP {response.status_code}"
        raise APIError(message=str(err_msg), status_code=response.status_code)


# ==================== Cartesia Sonic 超低延迟 TTS 适配器 ====================


class CartesiaAdapter(BaseAdapter):
    """
    Cartesia 适配器

    新一代超低延迟语音合成平台，Sonic 模型以实时性和自然度著称。

    API 规范：
    - Base URL: https://api.cartesia.ai
    - TTS 二进制: POST /v1/tts/bytes
    - TTS SSE流: POST /v1/tts/sse
    - 认证: X-API-Key header + Cartesia-Version: 2024-06-10
    - 文档: https://docs.cartesia.ai/api-reference/tts/bytes
    - 特点: 超低延迟（<200ms首包）、多语言、情感控制、声音克隆、实时流式
    """

    provider_type = "cartesia"
    provider_name = "Cartesia"
    supported_capabilities = [
        Capabilities.AUDIO_SPEECH,
    ]

    DEFAULT_BASE_URL = "https://api.cartesia.ai"
    DEFAULT_API_VERSION = "2024-06-10"

    DEFAULT_VOICES = {
        "Barbershop Man": "a0e99841-438c-4a64-b679-ae501e7d6091",
        "Miles": "b27adc08-7b68-4a70-8bd3-4d7f4a4f91e8",
        "Cali Bot": "a3d1d9d3-5de0-4c8b-8325-04f0e08d0e8a",
        "Customer Support Lady": "297b6c1f-3135-403e-a9c7-28a7a4dcd690",
        "Doc Brown": "f114a467-c40a-4db8-964d-8aca59832a2f",
        "Generic Man": "c8e59920-61aa-494c-8cfe-9730e0ce4c65",
        "Generic Woman": "b2b21a4e-7f85-4905-a50e-70c04fb7cd9e",
        "Helpdesk Woman": "70ca091e-a631-4b0c-a08b-9c930ffb0a8e",
        "Japanese Woman": "8f2d130a-3a73-4d4e-bab7-d0c999371c71",
        "Classy British Lady": "56553793-917c-41a4-a057-750be38b4b61",
        "Merchant": "50d0be50-c6b1-4b10-a4dd-e10588f01b06",
        "Movie Guy": "248be419-c632-4f23-adf1-5324ed7dbf1d",
        "New York Guy": "01d31f54-303d-480a-9c27-22b9c9f3c85e",
        "News Lady": "bf9923ea-7a3f-4f62-817d-c38d59f82f33",
        "Nurse": "660569f7-267b-4240-9b74-0a4bb4dce2e5",
        "Polite Man": "156fb8d2-335b-4950-9cb3-a2d33bef8830",
        "Salesman": "87748186-23bb-4158-a1eb-332911b0b706",
        "Southern Woman": "5d1b15b6-65c6-4f3c-a349-2f2fcc5c13d1",
        "The Don": "820a3788-2b37-4d21-847a-b65d8a68c99a",
        "The Laughing Guy": "0a7e6a33-0006-447c-a723-133a0a5b09c8",
        "Chemistry Professor": "694f9389-aac1-45b6-b726-9d9369183238",
        "Chinese Woman": "e90c66b3-51bd-4906-a51a-469152d083b1",
        "Sharon": "e00d0e5a-87e7-469a-9023-d510c6f5f970",
        "Competitive Podcaster": "8c4a4d43-d33d-4a80-a9a6-a79ce1e39a69",
    }

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self.api_key = config.api_key or ""
        self.api_version = (
            getattr(config, "api_version", None) or self.DEFAULT_API_VERSION
        )
        self._http_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """启动适配器，初始化 HTTP 客户端"""
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "X-API-Key": self.api_key,
                "Cartesia-Version": self.api_version,
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        """关闭适配器，释放 HTTP 客户端"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端"""
        if self._http_client is None:
            raise RuntimeError("Adapter not started. Call start() first.")
        return self._http_client

    def _get_voice_id(self, voice: str) -> str:
        """根据 voice 名称或ID 获取 Cartesia voice_id"""
        if voice in self.DEFAULT_VOICES:
            return self.DEFAULT_VOICES[voice]
        return voice

    async def speech(
        self,
        model: str,
        input: str,
        voice: str = "Generic Woman",
        **kwargs: Any,
    ) -> SpeechResult:
        """
        文字转语音 (Cartesia Sonic TTS)

        Args:
            model: 模型 ID (sonic-2 / sonic-english / sonic-multilingual)
            input: 要合成的文本
            voice: 音色名称或 voice_id
            **kwargs: 其他参数
                - output_format: dict 输出格式配置，默认 pcm_f32le 24kHz
                    {"container":"mp3","bit_rate":128000,"sample_rate":44100}
                    或 {"container":"wav","encoding":"pcm_f32le","sample_rate":24000}
                - language: str 语言代码 (en/zh/es/fr/de/ja/ko/pt 等)
                - speed: float 语速 (0.5-2.0)
                - emotion: list[str] 情感控制 ["positivity"]/["curiosity"]/["surprise"] 等
                - voice_id: str 直接指定voice_id（覆盖voice参数）
                - voice_embedding: list[float] 自定义音色embedding（声音克隆）

        Returns:
            SpeechResult
        """
        client = self._get_client()
        voice_id = kwargs.get("voice_id") or self._get_voice_id(voice)

        output_format = kwargs.get("output_format")
        if not output_format:
            output_format = {
                "container": "mp3",
                "bit_rate": 128000,
                "sample_rate": 44100,
            }

        payload: dict[str, Any] = {
            "model_id": model,
            "transcript": input,
            "voice": {"mode": "id", "id": voice_id},
            "output_format": output_format,
        }

        voice_embedding = kwargs.get("voice_embedding")
        if voice_embedding:
            payload["voice"] = {"mode": "embedding", "embedding": voice_embedding}

        language = kwargs.get("language")
        if language:
            payload["language"] = language

        if "speed" in kwargs:
            payload["speed"] = kwargs["speed"]

        emotion = kwargs.get("emotion")
        if emotion:
            payload["emotions"] = emotion if isinstance(emotion, list) else [emotion]

        if kwargs.get("continue") or kwargs.get("continuation"):
            payload["continue"] = True
        if kwargs.get("add_timestamps"):
            payload["add_timestamps"] = True

        accept_map = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "opus": "audio/ogg;codec=opus",
            "raw": "audio/pcm",
            "flac": "audio/flac",
        }
        container = (
            output_format.get("container", "mp3")
            if isinstance(output_format, dict)
            else "mp3"
        )
        accept_type = accept_map.get(container, "audio/mpeg")

        response = await client.post(
            "/v1/tts/bytes",
            json=payload,
            headers={"Accept": accept_type},
        )
        self._handle_error(response)

        fmt_ext = container
        content_type = accept_type

        return SpeechResult(
            audio_data=response.content,
            content_type=content_type,
            format=fmt_ext,
            model=model,
        )

    async def transcribe(
        self, model: str, file: Any, **kwargs: Any
    ) -> TranscriptionResult:
        """Cartesia 专注 TTS，不提供 ASR"""
        raise UnsupportedCapabilityError(
            message="Cartesia does not support speech-to-text (transcription)",
            details={"provider": self.provider_type, "capability": "audio_transcribe"},
        )

    async def chat(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> Any:
        raise UnsupportedCapabilityError(
            message="Cartesia does not support chat completions",
            details={"provider": self.provider_type, "capability": "chat"},
        )

    async def chat_stream(
        self, model: str, messages: list[ChatMessage], **kwargs: Any
    ) -> Any:
        if False:
            yield
        raise UnsupportedCapabilityError(
            message="Cartesia does not support streaming chat",
            details={"provider": self.provider_type, "capability": "chat_stream"},
        )

    async def image_generate(self, model: str, prompt: str, **kwargs: Any) -> Any:
        raise UnsupportedCapabilityError(
            message="Cartesia does not support image generation",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(self, model: str, prompt: str, **kwargs: Any) -> Any:
        raise UnsupportedCapabilityError(
            message="Cartesia does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(self, task_id: str, model: str = "") -> Any:
        raise UnsupportedCapabilityError(
            message="Cartesia does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(self, model_type: str | None = None) -> list[ModelInfo]:
        """列出 Cartesia 可用 TTS 模型"""
        models = [
            ModelInfo(
                id="sonic-2",
                name="Sonic 2",
                type="audio",
                provider="cartesia",
                capabilities=["audio_speech"],
                description="Sonic 2 最新模型，最佳质量和多语言支持",
            ),
            ModelInfo(
                id="sonic-2-2025-04-01",
                name="Sonic 2 (2025-04-01)",
                type="audio",
                provider="cartesia",
                capabilities=["audio_speech"],
                description="Sonic 2 固定版本",
            ),
            ModelInfo(
                id="sonic-turbo",
                name="Sonic Turbo",
                type="audio",
                provider="cartesia",
                capabilities=["audio_speech"],
                description="Sonic Turbo 超低延迟版本",
            ),
            ModelInfo(
                id="sonic-preview",
                name="Sonic Preview",
                type="audio",
                provider="cartesia",
                capabilities=["audio_speech"],
                description="Sonic 预览版",
            ),
        ]
        if model_type:
            models = [m for m in models if m.type == model_type]
        return models

    def _handle_error(self, response: httpx.Response) -> None:
        """处理 Cartesia 错误响应"""
        if response.status_code < 400:
            return
        if response.status_code == 401:
            raise AuthenticationError(message="Invalid Cartesia API key (X-API-Key)")
        if response.status_code == 429:
            raise RateLimitError(
                message="Cartesia rate limit exceeded or quota exhausted"
            )
        if response.status_code == 404:
            raise APIError(
                message="Cartesia voice_id or model not found", status_code=404
            )
        if response.status_code in (400, 422):
            try:
                error_data = response.json()
                err_msg = (
                    error_data.get("message")
                    or error_data.get("error")
                    or str(error_data)
                )
                if isinstance(err_msg, dict):
                    err_msg = err_msg.get("message", str(err_msg))
            except Exception:
                err_msg = f"Bad request: {response.text[:200]}"
            raise APIError(message=str(err_msg), status_code=response.status_code)
        try:
            error_data = response.json()
            err_msg = error_data.get("message") or error_data.get(
                "error", f"HTTP {response.status_code}"
            )
        except Exception:
            err_msg = f"HTTP {response.status_code}"
        raise APIError(message=str(err_msg), status_code=response.status_code)


# ==================== Edge TTS 免费神经语音合成适配器 ====================


class EdgeTTSAdapter(BaseAdapter):
    """
    Edge TTS 适配器

    基于 Microsoft Edge 浏览器的免费神经语音合成服务。
    底层使用微软 Azure 神经语音合成引擎，但不需要 API Key，完全免费。

    API 规范：
    - 基于 edge-tts Python 库（封装微软认知服务 WebSocket 协议）
    - 不需要 API Key，免费使用
    - 支持 100+ 种语音，覆盖 50+ 语言
    - 支持 MP3/WEBM/OGG 等输出格式
    - 支持语速、音调、音量调整
    - 文档: https://github.com/rany2/edge-tts
    - 特点: 完全免费、高质量神经语音、多语言、中文支持好
    """

    provider_type = "edge-tts"
    provider_name = "Edge TTS"
    supported_capabilities = [
        Capabilities.AUDIO_SPEECH,
    ]
    # Edge TTS 基于微软免费神经语音服务，无需 API Key 认证
    requires_api_key = False

    DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"

    COMMON_VOICES = {
        # 中文女声
        "xiaoxiao": "zh-CN-XiaoxiaoNeural",
        "晓晓": "zh-CN-XiaoxiaoNeural",
        "xiaoyi": "zh-CN-XiaoyiNeural",
        "晓伊": "zh-CN-XiaoyiNeural",
        "xiaochen": "zh-CN-XiaochenNeural",
        "晓辰": "zh-CN-XiaochenNeural",
        "xiaohan": "zh-CN-XiaohanNeural",
        "晓涵": "zh-CN-XiaohanNeural",
        "xiaomeng": "zh-CN-XiaomengNeural",
        "晓梦": "zh-CN-XiaomengNeural",
        "xiaomo": "zh-CN-XiaomoNeural",
        "晓墨": "zh-CN-XiaomoNeural",
        "xiaoqiu": "zh-CN-XiaoqiuNeural",
        "晓秋": "zh-CN-XiaoqiuNeural",
        "xiaorui": "zh-CN-XiaoruiNeural",
        "晓睿": "zh-CN-XiaoruiNeural",
        "xiaoshuang": "zh-CN-XiaoshuangNeural",
        "晓双": "zh-CN-XiaoshuangNeural",
        "xiaoxuan": "zh-CN-XiaoxuanNeural",
        "晓萱": "zh-CN-XiaoxuanNeural",
        "xiaoyan": "zh-CN-XiaoyanNeural",
        "晓颜": "zh-CN-XiaoyanNeural",
        "xiaoyou": "zh-CN-XiaoyouNeural",
        "晓悠": "zh-CN-XiaoyouNeural",
        # 中文男声
        "yunjian": "zh-CN-YunjianNeural",
        "云健": "zh-CN-YunjianNeural",
        "yunxi": "zh-CN-YunxiNeural",
        "云希": "zh-CN-YunxiNeural",
        "yunxia": "zh-CN-YunxiaNeural",
        "云夏": "zh-CN-YunxiaNeural",
        "yunyang": "zh-CN-YunyangNeural",
        "云扬": "zh-CN-YunyangNeural",
        "yunque": "zh-CN-YunzeNeural",
        "云泽": "zh-CN-YunzeNeural",
        # 英文女声
        "jenny": "en-US-JennyNeural",
        "jenny-multilingual": "en-US-JennyMultilingualNeural",
        "aria": "en-US-AriaNeural",
        "guy": "en-US-GuyNeural",
        "roger": "en-US-RogerNeural",
        # 英文男声
        "davis": "en-US-DavisNeural",
        "tony": "en-US-TonyNeural",
        "jason": "en-US-JasonNeural",
        # 日文
        "nanami": "ja-JP-NanamiNeural",
        "keita": "ja-JP-KeitaNeural",
        # 韩文
        "sun-hi": "ko-KR-SunHiNeural",
        "in-jun": "ko-KR-InJoonNeural",
        # 法文
        "denise": "fr-FR-DeniseNeural",
        "henri": "fr-FR-HenriNeural",
        # 德文
        "katja": "de-DE-KatjaNeural",
        "conrad": "de-DE-ConradNeural",
        # 西班牙文
        "elvira": "es-ES-ElviraNeural",
        "alvaro": "es-ES-AlvaroNeural",
    }

    OUTPUT_FORMATS = {
        "mp3": "audio-24khz-48kbit-mp3-mono",
        "mp3-96k": "audio-24khz-96kbit-mp3-mono",
        "mp3-128k": "audio-24khz-128kbit-mp3-mono",
        "mp3-160k": "audio-24khz-160kbit-mp3-mono",
        "webm": "audio-24khz-48kbit-opus-mono",
        "webm-24khz-16bit-mono-opus": "audio-24khz-16bit-mono-opus",
        "ogg": "audio-24khz-48kbit-opus-mono",
        "wav": "audio-24khz-16bit-mono-pcm",
        "pcm": "audio-24khz-16bit-mono-pcm",
    }

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._edge_tts_module: Any = None

    def _get_edge_tts(self) -> Any:
        """惰性导入 edge-tts 模块，未安装时给出清晰错误"""
        if self._edge_tts_module is None:
            try:
                import edge_tts

                self._edge_tts_module = edge_tts
            except ImportError:
                raise ImportError(
                    "edge-tts library not installed. "
                    "Install it with: pip install agn-sdk[edge-tts] "
                    "or pip install edge-tts"
                ) from None
        return self._edge_tts_module

    def _resolve_voice(self, voice: str) -> str:
        """解析语音名称，支持中文名、简称和完整 voice name"""
        if not voice:
            return self.DEFAULT_VOICE
        if voice in self.COMMON_VOICES:
            return self.COMMON_VOICES[voice]
        if voice.lower() in {k.lower() for k in self.COMMON_VOICES}:
            for k, v in self.COMMON_VOICES.items():
                if k.lower() == voice.lower():
                    return v
        if "-" in voice and voice.count("-") >= 2:
            return voice
        return self.DEFAULT_VOICE

    def _get_output_format(self, fmt: str | None) -> tuple[str, str]:
        """获取输出格式，返回 (edge-tts format string, content_type)"""
        fmt_key = (fmt or "mp3").lower()
        if fmt_key in self.OUTPUT_FORMATS:
            edge_fmt = self.OUTPUT_FORMATS[fmt_key]
        else:
            edge_fmt = fmt_key

        if "mp3" in edge_fmt:
            content_type = "audio/mpeg"
        elif "opus" in edge_fmt or "webm" in edge_fmt or "ogg" in edge_fmt:
            content_type = "audio/ogg"
        elif "pcm" in edge_fmt or "wav" in edge_fmt:
            content_type = "audio/wav"
        else:
            content_type = "audio/mpeg"

        return edge_fmt, content_type

    async def start(self) -> None:
        """启动适配器（验证 edge-tts 可用）"""
        self._get_edge_tts()

    async def close(self) -> None:
        """关闭适配器"""
        pass

    async def speech(
        self,
        model: str,
        input: str,
        voice: str = "",
        **kwargs: Any,
    ) -> SpeechResult:
        """
        文字转语音 (Edge TTS)

        Args:
            model: 模型标识（Edge TTS 只有一个模型，传入任意值都用 edge-tts）
            input: 要合成的文本
            voice: 音色名称或完整 voice ID
                支持简称: xiaoxiao/晓晓/yunxi/云希/jenny/nanami 等 40+ 预设
                支持完整ID: zh-CN-XiaoxiaoNeural
            **kwargs: 其他参数
                - rate: str 语速 "+0%" / "-20%" / "+50%"
                - pitch: str 音调 "+0Hz" / "+200Hz" / "-50Hz"
                - volume: str 音量 "+0%" / "-20%" / "+50%"
                - output_format: str 输出格式 (mp3/mp3-128k/webm/wav/ogg/pcm)
                - proxy: str 代理 URL
                - receive_timeout: int 接收超时秒数

        Returns:
            SpeechResult
        """
        edge_tts = self._get_edge_tts()
        voice_id = self._resolve_voice(voice)
        edge_fmt, content_type = self._get_output_format(kwargs.get("output_format"))

        communicate_kwargs: dict[str, Any] = {
            "text": input,
            "voice": voice_id,
        }

        if "rate" in kwargs:
            communicate_kwargs["rate"] = kwargs["rate"]
        if "pitch" in kwargs:
            communicate_kwargs["pitch"] = kwargs["pitch"]
        if "volume" in kwargs:
            communicate_kwargs["volume"] = kwargs["volume"]
        if "proxy" in kwargs:
            communicate_kwargs["proxy"] = kwargs["proxy"]
        if "receive_timeout" in kwargs:
            communicate_kwargs["receive_timeout"] = kwargs["receive_timeout"]

        communicate = edge_tts.Communicate(**communicate_kwargs)

        audio_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        audio_data = b"".join(audio_chunks)

        fmt_ext = "mp3"
        of = kwargs.get("output_format", "mp3")
        if isinstance(of, str):
            of_lower = of.lower()
            if "mp3" in of_lower:
                fmt_ext = "mp3"
            elif "webm" in of_lower:
                fmt_ext = "webm"
            elif "ogg" in of_lower:
                fmt_ext = "ogg"
            elif "wav" in of_lower or "pcm" in of_lower:
                fmt_ext = "wav"

        return SpeechResult(
            audio_data=audio_data,
            content_type=content_type,
            format=fmt_ext,
            model=model if model else "edge-tts",
        )

    async def transcribe(
        self, model: str, file: Any, **kwargs: Any
    ) -> TranscriptionResult:
        """Edge TTS 仅提供 TTS，不支持 ASR"""
        raise UnsupportedCapabilityError(
            message="Edge TTS does not support speech-to-text (transcription)",
            details={"provider": self.provider_type, "capability": "audio_transcribe"},
        )

    async def list_voices(self, language: str | None = None) -> list[dict[str, Any]]:
        """
        列出 Edge TTS 可用语音列表

        Args:
            language: 按语言过滤，如 "zh-CN" / "en-US" / "ja-JP"

        Returns:
            语音列表，每项包含 Name、ShortName、Locale、Gender 等
        """
        edge_tts = self._get_edge_tts()
        voices: list[dict[str, Any]] = await edge_tts.list_voices()
        if language:
            voices = [v for v in voices if v.get("Locale", "").startswith(language)]
        return voices

    async def chat(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> Any:
        raise UnsupportedCapabilityError(
            message="Edge TTS does not support chat completions",
            details={"provider": self.provider_type, "capability": "chat"},
        )

    async def chat_stream(
        self, model: str, messages: list[ChatMessage], **kwargs: Any
    ) -> Any:
        if False:
            yield
        raise UnsupportedCapabilityError(
            message="Edge TTS does not support streaming chat",
            details={"provider": self.provider_type, "capability": "chat_stream"},
        )

    async def image_generate(self, model: str, prompt: str, **kwargs: Any) -> Any:
        raise UnsupportedCapabilityError(
            message="Edge TTS does not support image generation",
            details={"provider": self.provider_type, "capability": "image"},
        )

    async def video_create(self, model: str, prompt: str, **kwargs: Any) -> Any:
        raise UnsupportedCapabilityError(
            message="Edge TTS does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def video_poll(self, task_id: str, model: str = "") -> Any:
        raise UnsupportedCapabilityError(
            message="Edge TTS does not support video generation",
            details={"provider": self.provider_type, "capability": "video"},
        )

    async def list_models(self, model_type: str | None = None) -> list[ModelInfo]:
        """列出 Edge TTS 模型"""
        models = [
            ModelInfo(
                id="edge-tts",
                name="Edge TTS Neural",
                type="audio",
                provider="edge-tts",
                capabilities=["audio_speech"],
                description="微软 Edge 浏览器免费神经语音合成，支持 100+ 种语音，50+ 语言，中文支持优秀",
            ),
        ]
        if model_type:
            models = [m for m in models if m.type == model_type]
        return models


# ==================== 注册适配器 ====================


AdapterFactory.register("elevenlabs", ElevenLabsAdapter)
AdapterFactory.register("eleven", ElevenLabsAdapter)
AdapterFactory.register("11labs", ElevenLabsAdapter)

AdapterFactory.register("deepgram", DeepgramAdapter)
AdapterFactory.register("dg", DeepgramAdapter)

AdapterFactory.register("assemblyai", AssemblyAIAdapter)
AdapterFactory.register("assembly", AssemblyAIAdapter)
AdapterFactory.register("aai", AssemblyAIAdapter)

AdapterFactory.register("cartesia", CartesiaAdapter)
AdapterFactory.register("sonic", CartesiaAdapter)

AdapterFactory.register("edge-tts", EdgeTTSAdapter)
AdapterFactory.register("edge_tts", EdgeTTSAdapter)
AdapterFactory.register("edge", EdgeTTSAdapter)
AdapterFactory.register("microsoft-tts", EdgeTTSAdapter)
