from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from RealtimeTTS.engines.breeze_tts_engine import (
    BREEZE_INFERENCE_REVISION,
    BREEZE_MODEL_REVISION,
    BreezeTTSEngine,
    BreezeTTSVoice,
    _BreezePyTorchBackend,
    _normalize_quantization,
)


class _FakeBackend:
    sample_rate = 24000

    def __init__(self, chunks=None):
        self.chunks = chunks or [np.array([-1.0, 0.0, 1.0], dtype=np.float32)]
        self.calls = []
        self.after_first = None
        self.shutdown_called = False

    def iter_audio_chunks(self, *, text, voice, seed):
        self.calls.append((text, voice, seed))
        for index, audio in enumerate(self.chunks):
            yield SimpleNamespace(
                audio=audio,
                sample_rate=self.sample_rate,
                timing={"decode_launch_ms": 1.0, "codec_launch_ms": 2.0},
            )
            if index == 0 and self.after_first is not None:
                self.after_first()

    def shutdown(self):
        self.shutdown_called = True


def _engine(backend, **kwargs):
    return BreezeTTSEngine(
        backend_factory=lambda **_factory_kwargs: backend,
        **kwargs,
    )


def test_streams_exact_pcm_and_records_latency_metrics():
    backend = _FakeBackend(
        chunks=[
            np.array([-1.0, -0.5, 0.0], dtype=np.float32),
            np.array([0.5, 1.0], dtype=np.float32),
        ]
    )
    voice = BreezeTTSVoice(
        name="calm",
        instruction="A calm, clear voice.",
        cfg_scale=4.0,
    )
    engine = _engine(backend, voice=voice, seed=7, quantization="q8")

    assert engine.synthesize("Hello from Breeze.") is True

    first = np.frombuffer(engine.queue.get_nowait(), dtype=np.int16)
    second = np.frombuffer(engine.queue.get_nowait(), dtype=np.int16)
    np.testing.assert_array_equal(first, np.array([-32767, -16383, 0], dtype=np.int16))
    np.testing.assert_array_equal(second, np.array([16383, 32767], dtype=np.int16))
    assert backend.calls == [("Hello from Breeze.", voice, 7)]
    assert engine.audio_duration == pytest.approx(5 / 24000)
    assert engine.last_synthesis_metrics["first_chunk_ms"] >= 0
    assert engine.last_synthesis_metrics["total_ms"] >= 0
    assert engine.last_synthesis_metrics["audio_seconds"] == pytest.approx(5 / 24000)
    assert engine.last_synthesis_metrics["quantization"] == "int8"
    assert engine.last_synthesis_metrics["timing_totals_ms"] == {
        "decode_launch_ms": 2.0,
        "codec_launch_ms": 4.0,
        "audio_d2h_ms": 0.0,
    }


def test_stop_event_discards_chunks_after_cancellation():
    backend = _FakeBackend(
        chunks=[
            np.array([0.25], dtype=np.float32),
            np.array([0.75], dtype=np.float32),
        ]
    )
    engine = _engine(backend)
    backend.after_first = engine.stop_synthesis_event.set

    assert engine.synthesize("Stop after one chunk.") is True

    assert engine.queue.qsize() == 1
    np.testing.assert_array_equal(
        np.frombuffer(engine.queue.get_nowait(), dtype=np.int16),
        np.array([8191], dtype=np.int16),
    )


def test_empty_backend_output_is_a_failure():
    backend = _FakeBackend()
    backend.chunks = []
    engine = _engine(backend)

    assert engine.synthesize("This must produce audio.") is False
    assert engine.queue.empty()
    assert engine.audio_duration == 0
    assert engine.last_synthesis_metrics["error"] == (
        "Breeze backend produced no audio"
    )


def test_upstream_revisions_are_pinned():
    assert len(BREEZE_INFERENCE_REVISION) == 40
    assert len(BREEZE_MODEL_REVISION) == 40


def test_voice_reference_requires_audio_and_exact_text_together(tmp_path):
    ref_audio = tmp_path / "reference.wav"
    ref_audio.write_bytes(b"RIFF")

    with pytest.raises(ValueError, match="together"):
        BreezeTTSVoice(ref_audio_path=str(ref_audio))
    with pytest.raises(ValueError, match="together"):
        BreezeTTSVoice(ref_text="Exact transcript")
    with pytest.raises(FileNotFoundError, match="reference audio"):
        BreezeTTSVoice(
            ref_audio_path=str(tmp_path / "missing.wav"),
            ref_text="Exact transcript",
        )


def test_quantization_aliases_and_fast_path_guard():
    assert _normalize_quantization(None) == "none"
    assert _normalize_quantization("Q8") == "int8"
    assert _normalize_quantization("8-bit") == "int8"
    assert _normalize_quantization("nf4") == "nf4"
    assert _normalize_quantization("Q4") == "nf4"
    with pytest.raises(ValueError, match="quantization"):
        _normalize_quantization("awq")

    with pytest.raises(ValueError, match="fast path"):
        _engine(_FakeBackend(), quantization="int8", fast_all=True)
    with pytest.raises(ValueError, match="fast path"):
        _engine(_FakeBackend(), quantization="nf4", fast_codec=True)


def test_bitsandbytes_configs_keep_runtime_cast_heads_unquantized():
    captured = []

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    backend = object.__new__(_BreezePyTorchBackend)
    backend.quantization = "int8"
    backend.llm_int8_threshold = 6.0
    backend.quantization_skip_modules = ()
    backend._quantization_config(
        FakeBitsAndBytesConfig,
        SimpleNamespace(float16="fp16"),
        "fp16",
    )
    assert captured[-1] == {
        "load_in_8bit": True,
        "llm_int8_threshold": 6.0,
        "llm_int8_skip_modules": ["lm_head", "codebooks_head"],
    }

    backend.quantization = "nf4"
    backend._quantization_config(
        FakeBitsAndBytesConfig,
        SimpleNamespace(float16="fp16"),
        "fp16",
    )
    assert captured[-1] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "fp16",
        "bnb_4bit_use_double_quant": True,
        "llm_int8_skip_modules": ["lm_head", "codebooks_head"],
    }

    backend.quantization = "int8"
    backend.llm_int8_threshold = 0.0
    backend.quantization_skip_modules = ("depth_decoder", "lm_head")
    backend._quantization_config(
        FakeBitsAndBytesConfig,
        SimpleNamespace(float16="fp16"),
        "fp16",
    )
    assert captured[-1] == {
        "load_in_8bit": True,
        "llm_int8_threshold": 0.0,
        "llm_int8_skip_modules": [
            "lm_head",
            "codebooks_head",
            "depth_decoder",
        ],
    }


def test_auto_dtype_uses_fp16_on_turing_and_bf16_on_ampere():
    class FakeCuda:
        capability = (7, 5)

        @classmethod
        def get_device_capability(cls, _device):
            return cls.capability

    fake_torch = SimpleNamespace(
        cuda=FakeCuda,
        float16="fp16",
        bfloat16="bf16",
    )
    backend = object.__new__(_BreezePyTorchBackend)
    backend.device = "cuda:0"

    assert backend._resolve_dtype(fake_torch, "auto") == "fp16"
    FakeCuda.capability = (8, 0)
    assert backend._resolve_dtype(fake_torch, "auto") == "bf16"


def test_attention_override_reaches_breeze_text_encoder_preference():
    model_config = SimpleNamespace(
        _attn_implementation="flash_attention_2",
        text_encoder_config=SimpleNamespace(
            _attn_implementation="flash_attention_2",
            preferred_attn_implementation="flash_attention_2",
        ),
    )

    _BreezePyTorchBackend._configure_attention(model_config, "eager")

    assert model_config._attn_implementation == "eager"
    assert model_config.text_encoder_config._attn_implementation == "eager"
    assert model_config.text_encoder_config.preferred_attn_implementation == "eager"


@pytest.mark.parametrize(
    ("quantization", "expected_class"),
    [("int8", "Linear8bitLt"), ("nf4", "Linear4bit")],
)
def test_real_bitsandbytes_replacement_skips_runtime_cast_heads(
    quantization,
    expected_class,
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    from transformers import BitsAndBytesConfig
    from transformers.integrations import replace_with_bnb_linear

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.lm_head = torch.nn.Linear(4, 3, bias=False)
            self.depth_decoder = torch.nn.Module()
            self.depth_decoder.codebooks_head = torch.nn.Linear(4, 3, bias=False)

    backend = object.__new__(_BreezePyTorchBackend)
    backend.quantization = quantization
    backend.llm_int8_threshold = 6.0
    backend.quantization_skip_modules = ()
    config = backend._quantization_config(
        BitsAndBytesConfig,
        torch,
        torch.float16,
    )
    model = TinyModel()
    model = replace_with_bnb_linear(
        model,
        modules_to_not_convert=["lm_head", "codebooks_head"],
        quantization_config=config,
    )

    assert type(model.encoder).__name__ == expected_class
    assert isinstance(model.lm_head, torch.nn.Linear)
    assert isinstance(model.depth_decoder.codebooks_head, torch.nn.Linear)


def test_voice_parameters_and_shutdown_are_forwarded():
    backend = _FakeBackend()
    engine = _engine(backend, voice="A warm narrator.")

    engine.set_voice_parameters(instruction="A brisk narrator.", cfg_scale=2.5)

    assert engine.voice.instruction == "A brisk narrator."
    assert engine.voice.cfg_scale == 2.5
    engine.shutdown()
    assert backend.shutdown_called is True


def test_public_exports_and_install_extra_are_declared():
    import RealtimeTTS
    import RealtimeTTS.engines

    assert RealtimeTTS.BreezeTTSEngine is BreezeTTSEngine
    assert RealtimeTTS.BreezeTTSVoice is BreezeTTSVoice
    assert RealtimeTTS.engines.BreezeTTSEngine is BreezeTTSEngine
    setup_text = (Path(__file__).parents[1] / "setup.py").read_text(encoding="utf-8")
    assert '"breeze": base_requirements + breeze_requirements' in setup_text
    assert "bitsandbytes>=0.50.1,<0.51" in setup_text
    assert "qwen-tts==0.1.1" in setup_text
