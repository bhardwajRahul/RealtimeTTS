"""Streaming client for Higgs Audio v3 served by SGLang-Omni."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

import requests

from .._audio_backend import pyaudio
from .base_engine import BaseEngine


_LOGGER = logging.getLogger(__name__)
_RESERVED_EXTRA_PAYLOAD_FIELDS = {"input", "stream", "response_format", "voice"}


@dataclass
class HiggsVoice:
    name: str = "default"
    prefix: str = ""

    def __repr__(self) -> str:
        if self.prefix:
            return f"HiggsVoice(name={self.name}, prefix={self.prefix!r})"
        return f"HiggsVoice(name={self.name})"


class HiggsEngine(BaseEngine):
    """Stream raw PCM from SGLang-Omni's OpenAI-compatible speech endpoint."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        voice: Optional[Union[str, HiggsVoice]] = None,
        control_prefix: str = "",
        model: Optional[str] = None,
        sample_rate: int = 24000,
        temperature: float = 0.8,
        top_k: int = 50,
        max_new_tokens: int = 1024,
        initial_codec_chunk_frames: Optional[int] = None,
        timeout: Optional[Union[float, tuple[float, float]]] = (10.0, 300.0),
        headers: Optional[Mapping[str, str]] = None,
        extra_payload: Optional[Mapping[str, Any]] = None,
        debug: bool = False,
    ) -> None:
        if int(sample_rate) <= 0:
            raise ValueError("Higgs sample_rate must be positive")
        if (
            initial_codec_chunk_frames is not None
            and int(initial_codec_chunk_frames) < 0
        ):
            raise ValueError("initial_codec_chunk_frames must be zero or greater")
        reserved = _RESERVED_EXTRA_PAYLOAD_FIELDS.intersection(extra_payload or {})
        if reserved:
            raise ValueError(
                "extra_payload cannot override reserved streaming fields: "
                + ", ".join(sorted(reserved))
            )

        self.api_url = (
            api_url
            or os.environ.get("HIGGS_TTS_API_URL")
            or "http://127.0.0.1:8000/v1/audio/speech"
        )
        self.model = model
        self.sample_rate = int(sample_rate)
        self.channels = 1
        self.sample_width = 2
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.initial_codec_chunk_frames = initial_codec_chunk_frames
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.extra_payload = dict(extra_payload or {})
        self.debug = debug
        self.session = requests.Session()

        self.current_voice = HiggsVoice(prefix=control_prefix)
        if voice is not None:
            self.set_voice(voice)

        self.last_first_audio_time: Optional[float] = None
        self.last_wall_time: Optional[float] = None
        self.last_chunk_count = 0
        self.last_error: Optional[BaseException] = None

    def post_init(self) -> None:
        self.engine_name = "higgs"
        self.can_consume_generators = False
        self.provides_word_timings = False

    def get_stream_info(self):
        return pyaudio.paInt16, self.channels, self.sample_rate

    def _build_payload(self, text: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "input": f"{self.current_voice.prefix}{text}",
            "voice": self.current_voice.name,
            "stream": True,
            "response_format": "pcm",
            "temperature": self.temperature,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.model:
            payload["model"] = self.model
        if self.initial_codec_chunk_frames is not None:
            payload["initial_codec_chunk_frames"] = int(
                self.initial_codec_chunk_frames
            )
        payload.update(self.extra_payload)
        return {key: value for key, value in payload.items() if value is not None}

    def _validate_response_headers(self, response: requests.Response) -> None:
        headers = {str(key).lower(): value for key, value in response.headers.items()}
        content_type = str(headers.get("content-type", "")).split(";", 1)[0].strip()
        if content_type and content_type not in {
            "audio/pcm",
            "application/octet-stream",
        }:
            raise ValueError(
                f"Higgs streaming response must be audio/pcm, got {content_type!r}"
            )

        expected = {
            "x-sample-rate": self.sample_rate,
            "x-channels": self.channels,
            "x-bit-depth": self.sample_width * 8,
        }
        for name, configured in expected.items():
            if name not in headers:
                continue
            try:
                actual = int(headers[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid Higgs response header {name}: {headers[name]!r}"
                ) from exc
            if actual != configured:
                raise ValueError(
                    f"Higgs response {name}={actual} "
                    f"does not match configured {configured}"
                )

    def _queue_pcm(self, pcm_bytes: bytes, started_at: float) -> None:
        if self.last_first_audio_time is None:
            self.last_first_audio_time = time.perf_counter() - started_at
        self.queue.put(pcm_bytes)
        frame_count = len(pcm_bytes) // (self.sample_width * self.channels)
        self.audio_duration += frame_count / float(self.sample_rate)
        self.last_chunk_count += 1

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)

        text = str(text or "").strip()
        if not text:
            self.last_error = ValueError("HiggsEngine text must not be empty")
            return False

        self.audio_duration = 0.0
        self.last_first_audio_time = None
        self.last_wall_time = None
        self.last_chunk_count = 0
        self.last_error = None
        started_at = time.perf_counter()

        try:
            with self.session.post(
                self.api_url,
                json=self._build_payload(text),
                headers=self.headers,
                stream=True,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                self._validate_response_headers(response)

                frame_size = self.sample_width * self.channels
                pending = b""
                for chunk in response.iter_content(chunk_size=None):
                    if self.stop_synthesis_event.is_set():
                        break
                    if not chunk:
                        continue
                    pending += bytes(chunk)
                    aligned_size = len(pending) - (len(pending) % frame_size)
                    if aligned_size:
                        self._queue_pcm(pending[:aligned_size], started_at)
                        pending = pending[aligned_size:]

                if pending and not self.stop_synthesis_event.is_set():
                    raise ValueError(
                        "Higgs PCM response ended with a partial audio frame"
                    )

            self.last_wall_time = time.perf_counter() - started_at
            if self.last_chunk_count or self.stop_synthesis_event.is_set():
                return True
            self.last_error = RuntimeError("Higgs server produced no audio")
            _LOGGER.error("Higgs synthesis failed: %s", self.last_error)
            return False
        except Exception as exc:
            self.last_wall_time = time.perf_counter() - started_at
            self.last_error = exc
            if self.debug:
                _LOGGER.exception("Higgs synthesis failed")
            else:
                _LOGGER.error("Higgs synthesis failed: %s", exc)
            return False

    def get_voices(self):
        return [HiggsVoice()]

    def set_voice(self, voice: Union[str, HiggsVoice]) -> None:
        if isinstance(voice, HiggsVoice):
            self.current_voice = voice
        else:
            self.current_voice = HiggsVoice(name=str(voice))

    def set_voice_parameters(self, **voice_parameters) -> None:
        for name in (
            "temperature",
            "top_k",
            "max_new_tokens",
            "initial_codec_chunk_frames",
            "timeout",
            "api_url",
            "model",
        ):
            if name in voice_parameters:
                setattr(self, name, voice_parameters[name])
        if "control_prefix" in voice_parameters:
            self.current_voice.prefix = str(
                voice_parameters["control_prefix"] or ""
            )
        if "voice" in voice_parameters:
            self.set_voice(voice_parameters["voice"])
        extra_payload = voice_parameters.get("extra_payload")
        if extra_payload:
            reserved = _RESERVED_EXTRA_PAYLOAD_FIELDS.intersection(extra_payload)
            if reserved:
                raise ValueError(
                    "extra_payload cannot override reserved streaming fields: "
                    + ", ".join(sorted(reserved))
                )
            self.extra_payload.update(dict(extra_payload))

    def shutdown(self) -> None:
        self.session.close()
