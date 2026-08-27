from __future__ import annotations

import re
import unicodedata
import warnings
from os import PathLike
from typing import Any, Sequence

import numpy as np

from .audio import wav_file_to_mono_float32
from .base import CharacterAlignment
from .vad import resample_audio


class TorchaudioMMSFACharacterAligner:
    """
    Character aligner backed by torchaudio's MMS forced-alignment bundle.
    """

    def __init__(
        self,
        device: str | None = None,
        dtype: Any = None,
    ):
        self.device = device
        self.dtype = dtype
        self._torch = None
        self._bundle = None
        self._model = None
        self._tokenizer = None
        self._aligner = None

    def align_characters(
        self,
        text: str,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Sequence[CharacterAlignment]:
        if audio.size == 0 or sample_rate <= 0:
            return []

        normalized_text = normalize_mms_fa_text(text)
        words = normalized_text.split()
        if not words:
            return []

        torch, bundle, model, tokenizer, aligner = self._load()
        model_audio = resample_audio(
            np.asarray(audio, dtype=np.float32),
            sample_rate,
            int(bundle.sample_rate),
        )
        if model_audio.size == 0:
            return []

        waveform = torch.from_numpy(model_audio).unsqueeze(0)
        if self.device is not None:
            waveform = waveform.to(self.device)
        if self.dtype is not None:
            waveform = waveform.to(dtype=self.dtype)

        with torch.inference_mode():
            emission, _ = model(waveform)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*forced_align has been deprecated.*",
                category=UserWarning,
            )
            spans_by_word = aligner(emission[0].float().cpu(), tokenizer(words))
        frame_duration = (len(audio) / float(sample_rate)) / float(emission.shape[1])
        alignments: list[CharacterAlignment] = []
        index = 0
        for word, spans in zip(words, spans_by_word):
            for character, span in zip(word, spans):
                alignments.append(
                    CharacterAlignment(
                        character=character,
                        start_time=float(span.start) * frame_duration,
                        end_time=float(span.end) * frame_duration,
                        index=index,
                        score=float(span.score),
                    )
                )
                index += 1
        return alignments

    def align_wav_characters(
        self,
        wav_path: str | PathLike,
        transcript: str,
        channel: int | None = None,
    ) -> Sequence[CharacterAlignment]:
        audio, sample_rate = wav_file_to_mono_float32(wav_path, channel=channel)
        return self.align_characters(transcript, audio, sample_rate)

    def _load(self):
        if (
            self._torch is not None
            and self._bundle is not None
            and self._model is not None
            and self._tokenizer is not None
            and self._aligner is not None
        ):
            return (
                self._torch,
                self._bundle,
                self._model,
                self._tokenizer,
                self._aligner,
            )

        try:
            import torch
            import torchaudio
        except ImportError as exc:
            raise ImportError(
                "MMS forced character alignment requires torch and torchaudio. "
                "Install them with `pip install realtimetts[alignment]`."
            ) from exc

        bundle = torchaudio.pipelines.MMS_FA
        model = bundle.get_model()
        if self.device is not None:
            model = model.to(self.device)
        if self.dtype is not None:
            model = model.to(dtype=self.dtype)
        model.eval()

        self._torch = torch
        self._bundle = bundle
        self._model = model
        self._tokenizer = bundle.get_tokenizer()
        self._aligner = bundle.get_aligner()
        return (
            self._torch,
            self._bundle,
            self._model,
            self._tokenizer,
            self._aligner,
        )


def normalize_mms_fa_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()
