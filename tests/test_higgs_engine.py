import pytest

from RealtimeTTS.engines.higgs_engine import HiggsEngine, HiggsVoice


class _FakeResponse:
    def __init__(self, chunks, headers=None, error=None):
        self.chunks = chunks
        self.headers = headers or {
            "Content-Type": "audio/pcm",
            "X-Sample-Rate": "24000",
            "X-Channels": "1",
            "X-Bit-Depth": "16",
        }
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        if self.error:
            raise self.error

    def iter_content(self, chunk_size=None):
        assert chunk_size is None
        yield from self.chunks


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response

    def close(self):
        self.closed = True


def _engine(monkeypatch, response, **kwargs):
    session = _FakeSession(response)
    monkeypatch.setattr(
        "RealtimeTTS.engines.higgs_engine.requests.Session",
        lambda: session,
    )
    return HiggsEngine(**kwargs), session


def test_higgs_streams_raw_pcm_and_builds_official_payload(monkeypatch):
    pcm = b"\x01\x00\x02\x00\x03\x00"
    response = _FakeResponse([pcm[:3], pcm[3:]])
    engine, session = _engine(
        monkeypatch,
        response,
        voice=HiggsVoice("narrator", prefix="<|emotion:calm|>"),
        model="bosonai/higgs-audio-v3-tts-4b",
        initial_codec_chunk_frames=20,
    )

    assert engine.synthesize("Hello") is True
    assert engine.queue.get_nowait() + engine.queue.get_nowait() == pcm
    assert engine.audio_duration == pytest.approx(3 / 24000)
    assert engine.last_chunk_count == 2
    assert engine.last_first_audio_time is not None
    url, request = session.calls[0]
    assert url.endswith("/v1/audio/speech")
    assert request["json"] == {
        "input": "<|emotion:calm|>Hello",
        "voice": "narrator",
        "stream": True,
        "response_format": "pcm",
        "temperature": 0.8,
        "top_k": 50,
        "max_new_tokens": 1024,
        "model": "bosonai/higgs-audio-v3-tts-4b",
        "initial_codec_chunk_frames": 20,
    }
    engine.shutdown()
    assert session.closed is True


def test_higgs_empty_response_is_failure(monkeypatch):
    engine, _session = _engine(monkeypatch, _FakeResponse([]))
    assert engine.synthesize("Hello") is False
    assert str(engine.last_error) == "Higgs server produced no audio"
    assert engine.queue.empty()


@pytest.mark.parametrize(
    "response",
    [
        _FakeResponse([b"\x01"], headers={"Content-Type": "audio/pcm"}),
        _FakeResponse(
            [b"\x01\x00"],
            headers={"Content-Type": "text/event-stream"},
        ),
        _FakeResponse(
            [b"\x01\x00"],
            headers={"Content-Type": "audio/pcm", "X-Sample-Rate": "44100"},
        ),
    ],
)
def test_higgs_rejects_invalid_stream_contract(monkeypatch, response):
    engine, _session = _engine(monkeypatch, response)
    assert engine.synthesize("Hello") is False
    assert engine.last_error is not None


def test_higgs_empty_text_and_reserved_payload_do_not_reach_server(monkeypatch):
    engine, session = _engine(monkeypatch, _FakeResponse([]))
    assert engine.synthesize("  \n ") is False
    assert session.calls == []
    with pytest.raises(ValueError, match="reserved"):
        _engine(
            monkeypatch,
            _FakeResponse([]),
            extra_payload={"response_format": "wav"},
        )


def test_higgs_public_lazy_exports():
    import RealtimeTTS
    import RealtimeTTS.engines

    assert RealtimeTTS.HiggsEngine is HiggsEngine
    assert RealtimeTTS.HiggsVoice is HiggsVoice
    assert RealtimeTTS.engines.HiggsEngine is HiggsEngine
    assert RealtimeTTS.engines.HiggsVoice is HiggsVoice
