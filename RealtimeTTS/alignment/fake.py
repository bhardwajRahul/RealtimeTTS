from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .base import WordAlignment


class FakeWordAligner:
    """
    Test helper that returns predeclared alignments without model downloads.
    """

    def __init__(self, alignments: Iterable[WordAlignment]):
        self.alignments = list(alignments)
        self.calls: List[tuple[str, int, int]] = []

    def align_words(
        self,
        text: str,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Sequence[WordAlignment]:
        self.calls.append((text, len(audio), sample_rate))
        return list(self.alignments)
