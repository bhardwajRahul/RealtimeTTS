"""Breeze TTS 2 engine for RealtimeTTS.

The upstream runtime is source-only and CUDA-only. This wrapper keeps imports
lazy, streams the upstream 24 kHz chunks into RealtimeTTS, and exposes optional
bitsandbytes INT8/NF4 loading for GPUs that cannot hold the BF16 checkpoint.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import numpy as np

from .base_engine import BaseEngine

_LOGGER = logging.getLogger(__name__)

# The adapter imports private modules from the upstream inference checkout, so
# both upstream inputs are pinned so compatibility can be reproduced and retested.
BREEZE_INFERENCE_REVISION = "ca632ce6c4d05f7985da4eab29b1a5d445b43f7b"
BREEZE_MODEL_REVISION = "c1c8ca18b70b30822735633991d9ebf4898e47d4"


_QUANTIZATION_ALIASES = {
    "": "none",
    "none": "none",
    "off": "none",
    "false": "none",
    "int8": "int8",
    "8bit": "int8",
    "8-bit": "int8",
    "q8": "int8",
    "q8_0": "int8",
    "nf4": "nf4",
    "int4": "nf4",
    "4bit": "nf4",
    "4-bit": "nf4",
    "q4": "nf4",
}


def _normalize_quantization(value: Optional[str]) -> str:
    normalized = str(value or "none").strip().lower().replace(" ", "")
    try:
        return _QUANTIZATION_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            "Breeze quantization must be one of: none, int8/q8, nf4/q4."
        ) from exc


class BreezeTTSVoice:
    """Voice design or reference-guided voice configuration."""

    def __init__(
        self,
        name: str = "default",
        instruction: str = "Speak clearly and naturally.",
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        cfg_scale: float = 1.0,
    ) -> None:
        instruction = str(instruction or "").strip()
        if not instruction:
            raise ValueError("Breeze voice instruction must not be empty.")
        if not math.isfinite(float(cfg_scale)) or float(cfg_scale) <= 0:
            raise ValueError("Breeze cfg_scale must be greater than 0.")

        ref_text = str(ref_text or "").strip() or None
        ref_audio_path = str(ref_audio_path) if ref_audio_path else None
        if bool(ref_audio_path) != bool(ref_text):
            raise ValueError(
                "Breeze ref_audio_path and exact ref_text must be provided together."
            )
        if ref_audio_path and not os.path.isfile(ref_audio_path):
            raise FileNotFoundError(
                f"Breeze reference audio was not found: {ref_audio_path}"
            )

        self.name = str(name or "default")
        self.instruction = instruction
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.cfg_scale = float(cfg_scale)

    def __repr__(self) -> str:
        mode = "reference" if self.ref_audio_path else "design"
        return (
            f"BreezeTTSVoice(name={self.name!r}, mode={mode!r}, "
            f"cfg_scale={self.cfg_scale!r})"
        )


class _BreezePyTorchBackend:
    """Thin adapter around the official Breeze source checkout."""

    _UNQUANTIZED_HEADS: ClassVar[tuple[str, str]] = (
        "lm_head",
        "codebooks_head",
    )

    def __init__(
        self,
        *,
        breeze_root: Optional[str],
        model_path: Optional[str],
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[str],
        token: Optional[str],
        device: str,
        dtype: str,
        quantization: str,
        llm_int8_threshold: float,
        quantization_skip_modules: tuple[str, ...],
        attn_implementation: str,
        max_new_tokens: int,
        max_seq_len: int,
        repetition_penalty: float,
        fast_all: Optional[bool],
        fast_text_encoder: bool,
        fast_backbone_prefill: bool,
        fast_backbone_decode: bool,
        fast_depth_decoder: bool,
        fast_codec: bool,
    ) -> None:
        self.device = device
        self.quantization = quantization
        self.llm_int8_threshold = float(llm_int8_threshold)
        self.quantization_skip_modules = quantization_skip_modules
        self._added_sys_path: Optional[str] = None
        self.torch = None
        self.model = None
        self.audio_tokenizer = None
        self.runtime = None
        self.tokenizer = None
        self.load_metrics: dict[str, Any] = {}

        if breeze_root:
            source_root = os.path.abspath(os.fspath(breeze_root))
            if not os.path.isdir(source_root):
                raise NotADirectoryError(
                    f"breeze_root is not a valid directory: {source_root}"
                )
            if source_root not in sys.path:
                sys.path.insert(0, source_root)
                self._added_sys_path = source_root

        try:
            self._load(
                model_path=model_path,
                model_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                dtype=dtype,
                attn_implementation=attn_implementation,
                max_new_tokens=max_new_tokens,
                max_seq_len=max_seq_len,
                repetition_penalty=repetition_penalty,
                fast_all=fast_all,
                fast_text_encoder=fast_text_encoder,
                fast_backbone_prefill=fast_backbone_prefill,
                fast_backbone_decode=fast_backbone_decode,
                fast_depth_decoder=fast_depth_decoder,
                fast_codec=fast_codec,
            )
        except Exception:
            self._remove_source_path()
            raise

    def _load(
        self,
        *,
        model_path: Optional[str],
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[str],
        token: Optional[str],
        dtype: str,
        attn_implementation: str,
        max_new_tokens: int,
        max_seq_len: int,
        repetition_penalty: float,
        fast_all: Optional[bool],
        fast_text_encoder: bool,
        fast_backbone_prefill: bool,
        fast_backbone_decode: bool,
        fast_depth_decoder: bool,
        fast_codec: bool,
    ) -> None:
        try:
            import torch
            from breeze_infer.runtime import (
                set_all_seeds,
                update_generation_config_for_breeze,
            )
            from breeze_infer.templates import get_template, prepare_inputs
            from models.breeze import BreezeForConditionalGeneration
            from models.fast_streaming import (
                FastBreezeStreamingRuntime,
                FastStreamingConfig,
            )
            from transformers import AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "Breeze TTS 2 dependencies or source modules are missing. Clone "
                "https://github.com/breezeblue-ai/breeze-tts, pass breeze_root, "
                "and install the realtimetts[breeze] extra. "
                f"Original error: {exc}"
            ) from exc

        if not self.device.startswith("cuda"):
            raise RuntimeError("The official Breeze streaming runtime requires CUDA.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Breeze requested CUDA, but torch.cuda.is_available() is false. "
                "Install a CUDA-enabled PyTorch wheel before realtimetts[breeze]."
            )

        self.torch = torch
        resolved_dtype = self._resolve_dtype(torch, dtype)
        checkpoint = self._resolve_checkpoint(
            model_path=model_path,
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )
        bundled_audio_tokenizer = checkpoint / "audio_tokenizer"
        if not bundled_audio_tokenizer.is_dir():
            raise FileNotFoundError(
                "Breeze checkpoint is missing its bundled audio_tokenizer directory: "
                f"{bundled_audio_tokenizer}"
            )

        load_started = time.perf_counter()
        model_config = BreezeForConditionalGeneration.config_class.from_pretrained(
            checkpoint,
            revision=revision,
            token=token,
        )
        self._configure_attention(model_config, attn_implementation)
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            revision=revision,
            token=token,
        )
        model_kwargs: dict[str, Any] = {
            "config": model_config,
            "dtype": resolved_dtype,
            "attn_implementation": attn_implementation,
            "low_cpu_mem_usage": True,
        }
        if self.quantization != "none":
            model_kwargs["quantization_config"] = self._quantization_config(
                BitsAndBytesConfig,
                torch,
                resolved_dtype,
            )
            model_kwargs["device_map"] = {"": self.device}

        self.model = BreezeForConditionalGeneration.from_pretrained(
            checkpoint,
            **model_kwargs,
        )
        if self.quantization == "none":
            self.model.to(self.device)
        self.model.eval()
        update_generation_config_for_breeze(self.model)

        try:
            from qwen_tts import Qwen3TTSTokenizer
        except (ImportError, SystemExit) as exc:
            raise ImportError(
                "Breeze's bundled codec requires qwen-tts==0.1.1 and the external "
                "SoX executable on PATH."
            ) from exc

        self.audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(bundled_audio_tokenizer),
            device_map=self.device,
        )
        codec_model = getattr(self.audio_tokenizer, "model", None)
        if codec_model is not None:
            codec_model.to(device=self.device, dtype=resolved_dtype).eval()

        config = FastStreamingConfig(
            max_new_tokens=int(max_new_tokens),
            max_seq_len=int(max_seq_len),
            fast_all=fast_all,
            fast_text_encoder=bool(fast_text_encoder),
            fast_backbone_prefill=bool(fast_backbone_prefill),
            fast_backbone_decode=bool(fast_backbone_decode),
            fast_depth_decoder=bool(fast_depth_decoder),
            fast_codec=bool(fast_codec),
            repetition_penalty=float(repetition_penalty),
        )
        self.runtime = FastBreezeStreamingRuntime(
            self.model,
            self.audio_tokenizer,
            config,
            tokenizer=self.tokenizer,
        )
        self.sample_rate = int(self.runtime.sample_rate)
        self._set_all_seeds = set_all_seeds
        self._get_template = get_template
        self._prepare_inputs = prepare_inputs

        torch.cuda.synchronize(self.device)
        self.load_metrics = {
            "load_seconds": time.perf_counter() - load_started,
            "quantization": self.quantization,
            "dtype": str(resolved_dtype).replace("torch.", ""),
            **self._cuda_memory_metrics(),
        }

    @staticmethod
    def _configure_attention(model_config: Any, implementation: str) -> None:
        """Override Breeze's nested FlashAttention preference as requested."""
        model_config._attn_implementation = implementation
        text_encoder_config = getattr(model_config, "text_encoder_config", None)
        if text_encoder_config is not None:
            text_encoder_config._attn_implementation = implementation
            text_encoder_config.preferred_attn_implementation = implementation

    def _resolve_dtype(self, torch: Any, value: str):
        normalized = str(value or "auto").strip().lower()
        if normalized == "auto":
            capability = torch.cuda.get_device_capability(self.device)
            return torch.bfloat16 if capability[0] >= 8 else torch.float16
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        raise ValueError("Breeze dtype must be auto, float16/fp16, or bfloat16/bf16.")

    def _resolve_checkpoint(
        self,
        *,
        model_path: Optional[str],
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[str],
        token: Optional[str],
    ) -> Path:
        if model_path:
            checkpoint = Path(model_path).expanduser().resolve()
            if not checkpoint.is_dir():
                raise NotADirectoryError(
                    f"Breeze model_path is not a directory: {checkpoint}"
                )
            return checkpoint

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface-hub is required to download Breeze TTS 2."
            ) from exc
        return Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )
        )

    def _quantization_config(
        self,
        config_class: Any,
        torch: Any,
        compute_dtype: Any,
    ) -> Any:
        skipped_modules = list(
            dict.fromkeys((*self._UNQUANTIZED_HEADS, *self.quantization_skip_modules))
        )
        common = {"llm_int8_skip_modules": skipped_modules}
        if self.quantization == "int8":
            return config_class(
                load_in_8bit=True,
                llm_int8_threshold=self.llm_int8_threshold,
                **common,
            )
        if self.quantization == "nf4":
            return config_class(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                **common,
            )
        raise AssertionError(f"Unexpected quantization mode: {self.quantization}")

    def iter_audio_chunks(
        self,
        *,
        text: str,
        voice: BreezeTTSVoice,
        seed: int,
    ):
        request_id = f"realtimetts-{uuid.uuid4().hex}"
        request = {
            "id": request_id,
            "text": text,
            "instruction": voice.instruction,
            "speaker": "S0",
        }
        template_name = "tts_instruction"
        if voice.ref_audio_path:
            request["ref_audio_path"] = voice.ref_audio_path
            request["ref_text"] = voice.ref_text
            template_name = "ref_edit_tata"

        self._set_all_seeds(int(seed))
        inputs = self._prepare_inputs(
            self.tokenizer,
            self.audio_tokenizer,
            self.model,
            [request],
            self._get_template(template_name),
            guidance_scale=voice.cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        yield from self.runtime.iter_audio_chunks(inputs, request_id=request_id)

    def _cuda_memory_metrics(self) -> dict[str, float]:
        if self.torch is None or not self.torch.cuda.is_available():
            return {}
        free_bytes, total_bytes = self.torch.cuda.mem_get_info(self.device)
        divisor = 1024.0 * 1024.0
        return {
            "cuda_allocated_mib": self.torch.cuda.memory_allocated(self.device)
            / divisor,
            "cuda_reserved_mib": self.torch.cuda.memory_reserved(self.device) / divisor,
            "cuda_free_mib": free_bytes / divisor,
            "cuda_total_mib": total_bytes / divisor,
        }

    def _remove_source_path(self) -> None:
        if self._added_sys_path and self._added_sys_path in sys.path:
            sys.path.remove(self._added_sys_path)
        self._added_sys_path = None

    def shutdown(self) -> None:
        self.runtime = None
        self.audio_tokenizer = None
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        self._remove_source_path()


class BreezeTTSEngine(BaseEngine):
    """Stream Breeze TTS 2 audio into RealtimeTTS."""

    def __init__(
        self,
        breeze_root: Optional[str] = None,
        model_path: Optional[str] = None,
        model_id: str = "BreezeBlue/Breeze-TTS-2",
        revision: Optional[str] = BREEZE_MODEL_REVISION,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        voice: Optional[Union[str, BreezeTTSVoice]] = None,
        device: str = "cuda:0",
        dtype: str = "auto",
        quantization: Optional[str] = "none",
        llm_int8_threshold: float = 6.0,
        quantization_skip_modules: Optional[list[str]] = None,
        attn_implementation: str = "eager",
        seed: int = 42,
        max_new_tokens: int = 1500,
        max_seq_len: int = 2048,
        repetition_penalty: float = 1.1,
        fast_all: Optional[bool] = None,
        fast_text_encoder: bool = False,
        fast_backbone_prefill: bool = False,
        fast_backbone_decode: bool = False,
        fast_depth_decoder: bool = False,
        fast_codec: bool = False,
        backend_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.quantization = _normalize_quantization(quantization)
        self._validate_fast_path(
            fast_all=fast_all,
            fast_text_encoder=fast_text_encoder,
            fast_backbone_prefill=fast_backbone_prefill,
            fast_backbone_decode=fast_backbone_decode,
            fast_depth_decoder=fast_depth_decoder,
            fast_codec=fast_codec,
        )
        if int(max_new_tokens) <= 0 or int(max_seq_len) <= 0:
            raise ValueError("Breeze max_new_tokens and max_seq_len must be positive.")
        if float(repetition_penalty) <= 0:
            raise ValueError("Breeze repetition_penalty must be greater than 0.")
        if (
            not math.isfinite(float(llm_int8_threshold))
            or float(llm_int8_threshold) < 0
        ):
            raise ValueError("Breeze llm_int8_threshold must be finite and >= 0.")
        normalized_skip_modules = tuple(
            str(module).strip()
            for module in (quantization_skip_modules or [])
            if str(module).strip()
        )

        self.device = device
        self.dtype = dtype
        self.seed = int(seed)
        self.last_synthesis_metrics: dict[str, Any] = {}
        self.voice: BreezeTTSVoice
        self.set_voice(voice or BreezeTTSVoice())

        factory = backend_factory or _BreezePyTorchBackend
        self._backend = factory(
            breeze_root=breeze_root,
            model_path=model_path,
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            device=device,
            dtype=dtype,
            quantization=self.quantization,
            llm_int8_threshold=float(llm_int8_threshold),
            quantization_skip_modules=normalized_skip_modules,
            attn_implementation=attn_implementation,
            max_new_tokens=int(max_new_tokens),
            max_seq_len=int(max_seq_len),
            repetition_penalty=float(repetition_penalty),
            fast_all=fast_all,
            fast_text_encoder=fast_text_encoder,
            fast_backbone_prefill=fast_backbone_prefill,
            fast_backbone_decode=fast_backbone_decode,
            fast_depth_decoder=fast_depth_decoder,
            fast_codec=fast_codec,
        )
        self.sampling_rate = int(getattr(self._backend, "sample_rate", 24000))
        self.load_metrics = dict(getattr(self._backend, "load_metrics", {}))

    def post_init(self) -> None:
        self.engine_name = "breeze-tts-2"

    def _validate_fast_path(self, **flags: Any) -> None:
        requested = flags.get("fast_all") is True or any(
            bool(value)
            for name, value in flags.items()
            if name != "fast_all"
        )
        if self.quantization != "none" and requested:
            raise ValueError(
                "Breeze's CUDA-graph fast path is incompatible with bitsandbytes "
                "quantization; disable all fast flags for int8/nf4."
            )

    def get_stream_info(self):
        from .._audio_backend import pyaudio

        return pyaudio.paInt16, 1, self.sampling_rate

    def get_voices(self):
        return []

    def set_voice(self, voice: Union[str, BreezeTTSVoice]) -> None:
        if isinstance(voice, str):
            voice = BreezeTTSVoice(name="designed", instruction=voice)
        if not isinstance(voice, BreezeTTSVoice):
            raise TypeError("voice must be a BreezeTTSVoice or instruction string.")
        self.voice = voice

    def set_voice_parameters(self, **voice_parameters: Any) -> None:
        allowed = {
            "name",
            "instruction",
            "ref_audio_path",
            "ref_text",
            "cfg_scale",
        }
        current = {
            "name": self.voice.name,
            "instruction": self.voice.instruction,
            "ref_audio_path": self.voice.ref_audio_path,
            "ref_text": self.voice.ref_text,
            "cfg_scale": self.voice.cfg_scale,
        }
        current.update(
            {key: value for key, value in voice_parameters.items() if key in allowed}
        )
        self.voice = BreezeTTSVoice(**current)
        if "seed" in voice_parameters:
            self.seed = int(voice_parameters["seed"])

    @staticmethod
    def _to_float_mono(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            audio = value.detach().float().cpu().numpy()
        else:
            audio = np.asarray(value, dtype=np.float32)
        audio = np.squeeze(audio)
        if audio.ndim > 1:
            audio = audio[0]
        return np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)

    def _start_cuda_metrics(self) -> dict[str, float]:
        torch = getattr(self._backend, "torch", None)
        if torch is None or not torch.cuda.is_available():
            return {}
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        divisor = 1024.0 * 1024.0
        return {
            "cuda_free_before_mib": free_bytes / divisor,
            "cuda_total_mib": total_bytes / divisor,
            "cuda_allocated_before_mib": torch.cuda.memory_allocated(self.device)
            / divisor,
        }

    def _finish_cuda_metrics(self) -> dict[str, float]:
        torch = getattr(self._backend, "torch", None)
        if torch is None or not torch.cuda.is_available():
            return {}
        torch.cuda.synchronize(self.device)
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        divisor = 1024.0 * 1024.0
        return {
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(self.device)
            / divisor,
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(self.device)
            / divisor,
            "cuda_free_after_mib": free_bytes / divisor,
        }

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)
        del sentence_count

        text = str(text or "").strip()
        if not text:
            return True

        started = time.perf_counter()
        first_chunk_ms: Optional[float] = None
        audio_samples = 0
        chunk_timings: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {
            "quantization": self.quantization,
            **self._start_cuda_metrics(),
        }
        try:
            for chunk in self._backend.iter_audio_chunks(
                text=text,
                voice=self.voice,
                seed=self.seed,
            ):
                if self.stop_synthesis_event.is_set():
                    break
                audio_value = getattr(chunk, "audio", chunk)
                chunk_timing = getattr(chunk, "timing", None)
                if chunk_timing:
                    chunk_timings.append(dict(chunk_timing))
                sample_rate = int(
                    getattr(chunk, "sample_rate", self.sampling_rate)
                )
                audio = self._to_float_mono(audio_value)
                if not audio.size:
                    continue
                if first_chunk_ms is None:
                    first_chunk_ms = (time.perf_counter() - started) * 1000.0
                self.sampling_rate = sample_rate
                audio_samples += int(audio.size)
                self.audio_duration += audio.size / float(sample_rate)
                self.queue.put((audio * 32767).astype(np.int16).tobytes())

            total_ms = (time.perf_counter() - started) * 1000.0
            audio_seconds = audio_samples / float(self.sampling_rate)
            metrics.update(
                {
                    "first_chunk_ms": first_chunk_ms,
                    "total_ms": total_ms,
                    "audio_seconds": audio_seconds,
                    "rtf": (total_ms / 1000.0 / audio_seconds)
                    if audio_seconds
                    else None,
                    "chunk_timings": chunk_timings,
                    "timing_totals_ms": {
                        field: sum(
                            float(timing.get(field, 0.0)) for timing in chunk_timings
                        )
                        for field in (
                            "decode_launch_ms",
                            "codec_launch_ms",
                            "audio_d2h_ms",
                        )
                    },
                    **self._finish_cuda_metrics(),
                }
            )
            self.last_synthesis_metrics = metrics
            if audio_samples:
                return True
            if self.stop_synthesis_event.is_set():
                return True
            metrics["error"] = "Breeze backend produced no audio"
            _LOGGER.error(metrics["error"])
            return False
        except Exception as exc:
            metrics.update(
                {
                    "first_chunk_ms": first_chunk_ms,
                    "total_ms": (time.perf_counter() - started) * 1000.0,
                    "error": str(exc),
                    **self._finish_cuda_metrics(),
                }
            )
            self.last_synthesis_metrics = metrics
            _LOGGER.exception("Breeze TTS 2 synthesis failed")
            return False

    def shutdown(self) -> None:
        shutdown = getattr(self._backend, "shutdown", None)
        if callable(shutdown):
            shutdown()


__all__ = [
    "BREEZE_INFERENCE_REVISION",
    "BREEZE_MODEL_REVISION",
    "BreezeTTSEngine",
    "BreezeTTSVoice",
]
