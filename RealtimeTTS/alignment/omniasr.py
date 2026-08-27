from __future__ import annotations

import re
import unicodedata
from os import PathLike
from typing import Any, Sequence

import numpy as np

from .audio import wav_file_to_mono_float32
from .base import CharacterAlignment, WordAlignment
from .ctc import (
    TokenSpan,
    ctc_forced_align,
    log_softmax,
    token_spans_to_character_alignments,
    token_spans_to_word_alignments,
)
from .vad import (
    EnergyVoiceActivityDetector,
    SileroVoiceActivityDetector,
    clamp_and_merge_speech_segments,
    create_voice_activity_detector,
    speech_segments_to_frame_mask,
)


class OmniASRCTCAligner:
    """
    Lazy forced-aligner backed by Meta's omniASR CTC 300M checkpoint.
    """

    MODEL_ID = "facebook/omniASR-CTC-300M"
    MODEL_CARD = "omniASR_CTC_300M"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        dtype: Any = None,
        use_vad: bool = False,
        vad_detector=None,
        vad_tolerance_seconds: float = 0.05,
        non_speech_token_penalty: float | None = -30.0,
        vad_supplemental_after_coarse_start: bool = True,
        vad_supplemental_start_margin_seconds: float = 0.2,
    ):
        if model_id != self.MODEL_ID:
            raise ValueError(
                "OmniASRCTCAligner only supports facebook/omniASR-CTC-300M"
            )
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.use_vad = bool(use_vad)
        self.vad_detector = vad_detector
        self.vad_tolerance_seconds = float(vad_tolerance_seconds)
        self.non_speech_token_penalty = non_speech_token_penalty
        self.vad_supplemental_after_coarse_start = bool(
            vad_supplemental_after_coarse_start
        )
        self.vad_supplemental_start_margin_seconds = float(
            vad_supplemental_start_margin_seconds
        )
        self._pipeline = None
        self._torch = None
        self._batch_layout_cls = None
        self._asr_model_cls = None

    def align_words(
        self,
        text: str,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Sequence[WordAlignment]:
        normalized_text, token_spans, frame_duration = self._align_token_spans(
            text,
            audio,
            sample_rate,
            use_vad=self.use_vad,
        )
        if not token_spans:
            return []

        words, ranges = self._word_token_ranges(normalized_text)
        return token_spans_to_word_alignments(
            token_spans,
            words,
            ranges,
            frame_duration,
        )

    def align_characters(
        self,
        text: str,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Sequence[CharacterAlignment]:
        normalized_text, token_spans, frame_duration = self._align_token_spans(
            text,
            audio,
            sample_rate,
            use_vad=self.use_vad,
        )
        if not token_spans:
            return []

        characters, ranges = self._character_token_ranges(normalized_text)
        return token_spans_to_character_alignments(
            token_spans,
            characters,
            ranges,
            frame_duration,
        )

    def align_wav_characters(
        self,
        wav_path: str | PathLike,
        transcript: str,
        channel: int | None = None,
        use_vad: bool | None = True,
        vad_tolerance_seconds: float | None = None,
    ) -> Sequence[CharacterAlignment]:
        audio, sample_rate = wav_file_to_mono_float32(wav_path, channel=channel)
        normalized_text, token_spans, frame_duration = self._align_token_spans(
            transcript,
            audio,
            sample_rate,
            use_vad=self.use_vad if use_vad is None else bool(use_vad),
            vad_tolerance_seconds=(
                self.vad_tolerance_seconds
                if vad_tolerance_seconds is None
                else float(vad_tolerance_seconds)
            ),
        )
        if not token_spans:
            return []

        characters, ranges = self._character_token_ranges(normalized_text)
        return token_spans_to_character_alignments(
            token_spans,
            characters,
            ranges,
            frame_duration,
        )

    def _align_token_spans(
        self,
        text: str,
        audio: np.ndarray,
        sample_rate: int,
        use_vad: bool | None = None,
        vad_tolerance_seconds: float | None = None,
    ) -> tuple[str, list[TokenSpan], float]:
        if audio.size == 0 or sample_rate <= 0:
            return "", [], 0.0

        normalized_text = normalize_omniasr_text(text)
        if not normalized_text:
            return "", [], 0.0

        pipeline = self._load_pipeline()
        target_ids = self._encode_text(normalized_text)
        if not target_ids:
            return normalized_text, [], 0.0

        logits = self._compute_logits(pipeline, audio, sample_rate)
        if logits.shape[0] == 0:
            return normalized_text, [], 0.0

        duration_seconds = len(audio) / float(sample_rate)
        frame_duration = duration_seconds / float(logits.shape[0])
        nonblank_frame_mask = self._speech_frame_mask(
            audio,
            sample_rate,
            logits.shape[0],
            frame_duration,
            use_vad=self.use_vad if use_vad is None else bool(use_vad),
            vad_tolerance_seconds=(
                self.vad_tolerance_seconds
                if vad_tolerance_seconds is None
                else float(vad_tolerance_seconds)
            ),
        )
        blank_id = self._blank_id(pipeline)
        log_probs = log_softmax(logits)
        if (
            nonblank_frame_mask is not None
            and self.vad_supplemental_after_coarse_start
            and (self.vad_detector is None)
        ):
            initial_token_spans = ctc_forced_align(
                log_probs,
                target_ids,
                blank_id,
                nonblank_frame_mask=nonblank_frame_mask,
                nonblank_forbidden_penalty=self.non_speech_token_penalty,
            )
            if initial_token_spans:
                coarse_start = initial_token_spans[0].start_frame * frame_duration
                supplemental_after = max(
                    0.0,
                    coarse_start - self.vad_supplemental_start_margin_seconds,
                )
                supplemental_mask = self._speech_frame_mask(
                    audio,
                    sample_rate,
                    logits.shape[0],
                    frame_duration,
                    use_vad=True,
                    vad_tolerance_seconds=(
                        self.vad_tolerance_seconds
                        if vad_tolerance_seconds is None
                        else float(vad_tolerance_seconds)
                    ),
                    supplemental_after_seconds=supplemental_after,
                )
                if supplemental_mask is not None:
                    nonblank_frame_mask = supplemental_mask

        token_spans = ctc_forced_align(
            log_probs,
            target_ids,
            blank_id,
            nonblank_frame_mask=nonblank_frame_mask,
            nonblank_forbidden_penalty=self.non_speech_token_penalty,
        )
        return normalized_text, token_spans, frame_duration

    def _speech_frame_mask(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_count: int,
        frame_duration: float,
        use_vad: bool,
        vad_tolerance_seconds: float,
        supplemental_after_seconds: float | None = None,
    ) -> np.ndarray | None:
        if not use_vad:
            return None

        detector = create_voice_activity_detector(self.vad_detector)
        duration_seconds = len(audio) / float(sample_rate)
        segments = list(detector.detect_speech(audio, sample_rate))
        if supplemental_after_seconds is not None:
            segments.extend(
                self._supplemental_speech_segments(
                    audio,
                    sample_rate,
                    supplemental_after_seconds,
                )
            )
        padded_segments = clamp_and_merge_speech_segments(
            segments,
            duration_seconds,
            tolerance_seconds=vad_tolerance_seconds,
        )
        if not padded_segments:
            return None

        mask = speech_segments_to_frame_mask(
            padded_segments,
            frame_count,
            frame_duration,
        )
        if not mask.any():
            return None
        return mask

    def _supplemental_speech_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        after_seconds: float,
    ):
        segments = []
        segments.extend(EnergyVoiceActivityDetector().detect_speech(audio, sample_rate))
        try:
            low_threshold_detector = SileroVoiceActivityDetector(
                threshold=0.15,
                min_speech_duration_ms=40,
                min_silence_duration_ms=50,
                energy_fallback=False,
            )
            segments.extend(low_threshold_detector.detect_speech(audio, sample_rate))
        except ImportError:
            pass

        return [
            segment
            for segment in segments
            if float(segment.end_time) >= float(after_seconds)
        ]

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch
            from fairseq2.nn.batch_layout import BatchLayout
            from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        except ImportError as exc:
            raise ImportError(
                "CTC word alignment requires the optional omniASR dependencies. "
                "Install them with `pip install realtimetts[omniasr]`. "
                "On Windows/Python 3.12 this may be blocked by upstream "
                "omnilingual-asr/fairseq2 packaging; use WSL/Linux or an ONNX "
                "backend exported from facebook/omniASR-CTC-300M."
            ) from exc

        dtype = self.dtype if self.dtype is not None else torch.float32
        self._pipeline = ASRInferencePipeline(
            model_card=self.MODEL_CARD,
            device=self.device,
            dtype=dtype,
        )
        if not isinstance(self._pipeline.model, Wav2Vec2AsrModel):
            raise RuntimeError(
                "Loaded omniASR model is not a CTC Wav2Vec2 ASR model."
            )

        self._torch = torch
        self._batch_layout_cls = BatchLayout
        self._asr_model_cls = Wav2Vec2AsrModel
        return self._pipeline

    def _compute_logits(self, pipeline, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        torch = self._torch
        batch_layout_cls = self._batch_layout_cls
        assert torch is not None
        assert batch_layout_cls is not None

        audio_input = [{"waveform": audio.astype(np.float32), "sample_rate": sample_rate}]
        wavs = list(pipeline._build_audio_wavform_pipeline(audio_input).and_return())
        batch = pipeline._create_batch_simple([(wavs[0], None)])
        batch_layout = batch_layout_cls(
            batch.source_seqs.shape,
            seq_lens=batch.source_seq_lens,
            device=batch.source_seqs.device,
        )

        with torch.inference_mode():
            logits, layout = pipeline.model(batch.source_seqs, batch_layout)

        frame_count = int(layout.seq_lens[0])
        return logits[0, :frame_count].float().cpu().numpy()

    def _encode_text(self, text: str) -> list[int]:
        pipeline = self._load_pipeline()
        token_tensor = pipeline.token_encoder(text)
        if hasattr(token_tensor, "tolist"):
            token_ids = [int(token_id) for token_id in token_tensor.tolist()]
        else:
            token_ids = [int(token_id) for token_id in token_tensor]

        special_ids = self._special_ids(pipeline)
        return [token_id for token_id in token_ids if token_id not in special_ids]

    def _word_token_ranges(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        words: list[str] = []
        ranges: list[tuple[int, int]] = []
        for match in re.finditer(r"\S+", text):
            prefix_ids = self._encode_text(text[: match.start()])
            word_prefix_ids = self._encode_text(text[: match.end()])
            start_index = len(prefix_ids)
            end_index = len(word_prefix_ids)
            if end_index > start_index:
                words.append(match.group(0))
                ranges.append((start_index, end_index))
        return words, ranges

    def _character_token_ranges(
        self,
        text: str,
    ) -> tuple[list[str], list[tuple[int, int]]]:
        characters: list[str] = []
        ranges: list[tuple[int, int]] = []
        for index, character in enumerate(text):
            prefix_ids = self._encode_text(text[:index])
            character_prefix_ids = self._encode_text(text[: index + 1])
            start_index = len(prefix_ids)
            end_index = len(character_prefix_ids)
            if end_index > start_index:
                characters.append(character)
                ranges.append((start_index, end_index))
        return characters, ranges

    @staticmethod
    def _blank_id(pipeline) -> int:
        vocab_info = pipeline.tokenizer.vocab_info
        return int(getattr(vocab_info, "pad_idx", 0) or 0)

    @staticmethod
    def _special_ids(pipeline) -> set[int]:
        vocab_info = pipeline.tokenizer.vocab_info
        ids = set()
        for attr in ("pad_idx", "bos_idx", "eos_idx"):
            token_id = getattr(vocab_info, attr, None)
            if token_id is not None:
                ids.add(int(token_id))
        return ids


def normalize_omniasr_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()
