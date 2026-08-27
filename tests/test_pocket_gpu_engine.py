import sys
import types
from pathlib import Path

import numpy as np

from RealtimeTTS import PocketTTSGpuEngine, PocketTTSGpuVoice


def test_pocket_gpu_engine_streams_audio(monkeypatch):
    class FakeModel:
        sample_rate = 24000

        def generate_audio_stream(self, voice_state, text, **kwargs):
            assert voice_state == "fake-state"
            assert text == "hello"
            assert kwargs["teacher_forcing"] is True
            assert kwargs["frames_after_eos"] == 2
            yield np.array([0.0, 0.5, -0.5], dtype=np.float32)

    def fake_load_model(self):
        self.model = FakeModel()
        self.sample_rate = FakeModel.sample_rate

    def fake_set_voice(self, voice):
        self.current_voice = PocketTTSGpuVoice(str(voice))
        self.current_voice_state = "fake-state"

    monkeypatch.setattr(PocketTTSGpuEngine, "_load_model", fake_load_model)
    monkeypatch.setattr(PocketTTSGpuEngine, "set_voice", fake_set_voice)

    engine = PocketTTSGpuEngine(
        voice="alba",
        teacher_forcing=True,
        frames_after_eos=2,
    )
    try:
        assert engine.engine_name == "pocket_tts_gpu"
        assert engine.synthesize("hello") is True
        assert engine.audio_duration == 3 / 24000
        assert engine.queue.get_nowait() == np.array(
            [0, 16383, -16383], dtype=np.int16
        ).tobytes()
    finally:
        engine.shutdown()


def test_pocket_gpu_voice_representation():
    voice = PocketTTSGpuVoice("demo", audio_prompt_path="voice.wav")

    assert "voice.wav" in repr(voice)


def test_pocket_gpu_load_voice_state_preserves_dotted_module_names(monkeypatch):
    class FakeTensor:
        def to(self, _device):
            return self

    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = FakeTensor
    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors.__path__ = []
    fake_safetensors_torch = types.ModuleType("safetensors.torch")
    fake_safetensors_torch.load_file = lambda path, device: {
        "transformer.layers.0.self_attn.cache": FakeTensor(),
        "transformer.layers.0.self_attn.current_end": FakeTensor(),
    }
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)

    engine = object.__new__(PocketTTSGpuEngine)
    engine.device = "cpu"
    state = engine._load_voice_state(Path("fake.safetensors"))

    assert set(state) == {"transformer.layers.0.self_attn"}
    assert set(state["transformer.layers.0.self_attn"]) == {
        "cache",
        "current_end",
    }


def test_pocket_gpu_stamps_stateful_module_names(monkeypatch):
    class FakeStatefulModule:
        pass

    stateful_module = types.ModuleType("pocket_tts.modules.stateful_module")
    stateful_module.StatefulModule = FakeStatefulModule
    pocket_tts = types.ModuleType("pocket_tts")
    pocket_tts.__path__ = []
    modules = types.ModuleType("pocket_tts.modules")
    modules.__path__ = []
    monkeypatch.setitem(sys.modules, "pocket_tts", pocket_tts)
    monkeypatch.setitem(sys.modules, "pocket_tts.modules", modules)
    monkeypatch.setitem(
        sys.modules,
        "pocket_tts.modules.stateful_module",
        stateful_module,
    )

    stateful = FakeStatefulModule()

    class FakeRoot:
        def named_modules(self):
            yield "", object()
            yield "transformer.layers.0.self_attn", stateful

    engine = object.__new__(PocketTTSGpuEngine)
    engine.model = types.SimpleNamespace(flow_lm=FakeRoot(), mimi=FakeRoot())
    engine._stamp_stateful_module_names()

    assert stateful._module_absolute_name == "transformer.layers.0.self_attn"
