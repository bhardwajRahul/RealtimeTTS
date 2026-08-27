import wave

import numpy as np

from RealtimeTTS.alignment import (
    CharacterAlignment,
    SpeechSegment,
    WavTranscriptCharacterAligner,
)
from RealtimeTTS.alignment.audio import wav_file_to_mono_float32
from RealtimeTTS.alignment.ctc import (
    ctc_forced_align,
    log_softmax,
    token_spans_to_character_alignments,
)
from RealtimeTTS.alignment.torchaudio_fa import normalize_mms_fa_text
from RealtimeTTS.alignment.vad import (
    clamp_and_merge_speech_segments,
    speech_segments_to_frame_mask,
)


def _logits_for_path(path, vocab_size):
    logits = np.full((len(path), vocab_size), -8.0, dtype=np.float32)
    for frame, token_id in enumerate(path):
        logits[frame, token_id] = 8.0
    return logits


def test_token_spans_to_character_alignments_maps_each_character():
    blank = 0
    h = 1
    i = 2
    space = 3
    logits = _logits_for_path(
        [blank, h, h, blank, i, blank, space, blank],
        vocab_size=4,
    )

    token_spans = ctc_forced_align(log_softmax(logits), [h, i, space], blank)
    alignments = token_spans_to_character_alignments(
        token_spans,
        ["h", "i", " "],
        [(0, 1), (1, 2), (2, 3)],
        frame_duration_seconds=0.02,
    )

    assert [
        (item.character, item.index, item.start_time, item.end_time)
        for item in alignments
    ] == [
        ("h", 0, 0.02, 0.06),
        ("i", 1, 0.08, 0.1),
        (" ", 2, 0.12, 0.14),
    ]
    assert all(item.score is not None for item in alignments)


def test_ctc_nonblank_frame_mask_moves_text_out_of_silence():
    blank = 0
    a = 1
    logits = np.array(
        [
            [0.0, 8.0],
            [8.0, 0.0],
            [8.0, 0.0],
            [0.0, 7.0],
            [8.0, 0.0],
        ],
        dtype=np.float32,
    )

    unmasked = ctc_forced_align(log_softmax(logits), [a], blank)
    masked = ctc_forced_align(
        log_softmax(logits),
        [a],
        blank,
        nonblank_frame_mask=[False, False, False, True, True],
    )

    assert unmasked[0].start_frame == 0
    assert masked[0].start_frame == 3


def test_vad_tolerance_expands_speech_frame_mask():
    segments = clamp_and_merge_speech_segments(
        [SpeechSegment(0.10, 0.20)],
        duration_seconds=0.40,
        tolerance_seconds=0.05,
    )

    mask = speech_segments_to_frame_mask(
        segments,
        frame_count=20,
        frame_duration_seconds=0.02,
    )

    assert mask[:2].tolist() == [False, False]
    assert mask[2:13].all()
    assert not mask[13:].any()


def test_normalize_mms_fa_text_keeps_supported_characters():
    assert normalize_mms_fa_text("Okay. Yep. That's it. Mm-hmm.") == (
        "okay yep that's it mm hmm"
    )


def test_wav_file_to_mono_float32_reads_pcm16(tmp_path):
    wav_path = tmp_path / "stereo.wav"
    samples = np.array(
        [
            [0, 32767],
            [-32768, 0],
        ],
        dtype=np.int16,
    )
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(samples.tobytes())

    audio, sample_rate = wav_file_to_mono_float32(wav_path)

    assert sample_rate == 8000
    assert audio.dtype == np.float32
    assert np.allclose(audio, [0.49998474, -0.5], atol=1e-6)


def test_wav_file_to_mono_float32_can_select_channel(tmp_path):
    wav_path = tmp_path / "stereo.wav"
    samples = np.array(
        [
            [0, 32767],
            [-32768, 0],
        ],
        dtype=np.int16,
    )
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(samples.tobytes())

    left, sample_rate = wav_file_to_mono_float32(wav_path, channel=0)
    right, _ = wav_file_to_mono_float32(wav_path, channel=1)

    assert sample_rate == 8000
    assert np.allclose(left, [0.0, -1.0], atol=1e-6)
    assert np.allclose(right, [0.9999695, 0.0], atol=1e-6)


def test_wav_transcript_character_aligner_passes_audio_to_aligner(tmp_path):
    wav_path = tmp_path / "mono.wav"
    samples = np.array([0, 32767, -32768], dtype=np.int16)
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(samples.tobytes())

    fake_aligner = _FakeCharacterAligner()
    result = WavTranscriptCharacterAligner(fake_aligner).align_file(
        wav_path,
        "Hi",
        channel=0,
    )

    assert result == [CharacterAlignment("h", 0.0, 0.1, 0, 0.99)]
    assert fake_aligner.text == "Hi"
    assert fake_aligner.sample_rate == 16000
    assert np.allclose(fake_aligner.audio, [0.0, 0.9999695, -1.0], atol=1e-6)


class _FakeCharacterAligner:
    def align_characters(self, text, audio, sample_rate):
        self.text = text
        self.audio = audio
        self.sample_rate = sample_rate
        return [CharacterAlignment("h", 0.0, 0.1, 0, 0.99)]
