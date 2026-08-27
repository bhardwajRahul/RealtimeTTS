import numpy as np

from RealtimeTTS.alignment.ctc import ctc_forced_align, log_softmax
from RealtimeTTS.alignment.omniasr import OmniASRCTCAligner


def _logits_for_path(path, vocab_size):
    logits = np.full((len(path), vocab_size), -8.0, dtype=np.float32)
    for frame, token_id in enumerate(path):
        logits[frame, token_id] = 8.0
    return logits


def test_ctc_forced_align_returns_token_spans():
    blank = 0
    a = 1
    b = 2
    logits = _logits_for_path([blank, a, a, blank, b, blank], vocab_size=3)

    spans = ctc_forced_align(log_softmax(logits), [a, b], blank_id=blank)

    assert [(span.token_id, span.start_frame, span.end_frame) for span in spans] == [
        (a, 1, 3),
        (b, 4, 5),
    ]


def test_ctc_forced_align_handles_repeated_tokens():
    blank = 0
    a = 1
    logits = _logits_for_path([blank, a, blank, a, blank], vocab_size=2)

    spans = ctc_forced_align(log_softmax(logits), [a, a], blank_id=blank)

    assert [(span.token_id, span.start_frame, span.end_frame) for span in spans] == [
        (a, 1, 2),
        (a, 3, 4),
    ]


def test_omniasr_backend_is_pinned_to_requested_model():
    aligner = OmniASRCTCAligner()

    assert aligner.model_id == "facebook/omniASR-CTC-300M"
