from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .base import SpeechSegment, VoiceActivityDetector


class SileroVoiceActivityDetector:
    """
    Voice activity detector backed by Silero VAD.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 80,
        min_silence_duration_ms: int = 50,
        energy_fallback: bool = False,
        model_sample_rate: int = 16000,
    ):
        self.threshold = float(threshold)
        self.min_speech_duration_ms = int(min_speech_duration_ms)
        self.min_silence_duration_ms = int(min_silence_duration_ms)
        self.energy_fallback = bool(energy_fallback)
        self.model_sample_rate = int(model_sample_rate)
        self._model = None
        self._get_speech_timestamps = None

    def detect_speech(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Sequence[SpeechSegment]:
        if audio.size == 0 or sample_rate <= 0:
            return []

        model, get_speech_timestamps = self._load()
        model_audio = resample_audio(
            np.asarray(audio, dtype=np.float32),
            sample_rate,
            self.model_sample_rate,
        )
        if model_audio.size == 0:
            return []

        import torch

        timestamps = get_speech_timestamps(
            torch.from_numpy(model_audio),
            model,
            threshold=self.threshold,
            sampling_rate=self.model_sample_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=0,
        )

        segments = [
            SpeechSegment(
                start_time=float(item["start"]) / float(self.model_sample_rate),
                end_time=float(item["end"]) / float(self.model_sample_rate),
            )
            for item in timestamps
            if float(item["end"]) > float(item["start"])
        ]
        if self.energy_fallback:
            segments.extend(EnergyVoiceActivityDetector().detect_speech(audio, sample_rate))

        duration_seconds = len(audio) / float(sample_rate)
        return clamp_and_merge_speech_segments(segments, duration_seconds, 0.0)

    def _load(self):
        if self._model is not None and self._get_speech_timestamps is not None:
            return self._model, self._get_speech_timestamps

        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as exc:
            raise ImportError(
                "VAD-gated character alignment requires Silero VAD. "
                "Install it with `pip install silero-vad` or "
                "`pip install realtimetts[omniasr]`."
            ) from exc

        self._model = load_silero_vad()
        self._get_speech_timestamps = get_speech_timestamps
        return self._model, self._get_speech_timestamps


class EnergyVoiceActivityDetector:
    """
    Lightweight fallback VAD for tests and constrained environments.
    """

    def __init__(
        self,
        frame_duration_seconds: float = 0.02,
        hop_duration_seconds: float = 0.01,
        threshold_ratio: float = 0.12,
        min_speech_duration_seconds: float = 0.04,
        min_silence_duration_seconds: float = 0.05,
    ):
        self.frame_duration_seconds = float(frame_duration_seconds)
        self.hop_duration_seconds = float(hop_duration_seconds)
        self.threshold_ratio = float(threshold_ratio)
        self.min_speech_duration_seconds = float(min_speech_duration_seconds)
        self.min_silence_duration_seconds = float(min_silence_duration_seconds)

    def detect_speech(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Sequence[SpeechSegment]:
        if audio.size == 0 or sample_rate <= 0:
            return []

        audio = np.asarray(audio, dtype=np.float32)
        frame_size = max(1, int(round(self.frame_duration_seconds * sample_rate)))
        hop_size = max(1, int(round(self.hop_duration_seconds * sample_rate)))
        if audio.size < frame_size:
            rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
            if rms > 0.0:
                return [SpeechSegment(0.0, len(audio) / float(sample_rate))]
            return []

        energies = []
        for start in range(0, audio.size - frame_size + 1, hop_size):
            frame = audio[start : start + frame_size]
            energies.append(float(np.sqrt(np.mean(np.square(frame)))))
        if not energies:
            return []

        peak = max(energies)
        if peak <= 0.0:
            return []
        threshold = peak * self.threshold_ratio
        active = [energy >= threshold for energy in energies]
        segments = _active_frames_to_segments(
            active,
            hop_size / float(sample_rate),
            frame_size / float(sample_rate),
        )
        duration_seconds = len(audio) / float(sample_rate)
        return clamp_and_merge_speech_segments(
            segments,
            duration_seconds,
            tolerance_seconds=0.0,
            min_duration_seconds=self.min_speech_duration_seconds,
            merge_gap_seconds=self.min_silence_duration_seconds,
        )


def create_voice_activity_detector(
    detector: VoiceActivityDetector | None = None,
) -> VoiceActivityDetector:
    if detector is not None:
        return detector
    return SileroVoiceActivityDetector()


def resample_audio(
    audio: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if source_sample_rate == target_sample_rate:
        return audio
    if audio.size == 0:
        return audio

    try:
        import torch
        import torchaudio.functional as F

        resampled = F.resample(
            torch.from_numpy(audio),
            orig_freq=source_sample_rate,
            new_freq=target_sample_rate,
        )
        return resampled.cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        pass

    try:
        from scipy.signal import resample_poly

        gcd = math.gcd(source_sample_rate, target_sample_rate)
        return resample_poly(
            audio,
            target_sample_rate // gcd,
            source_sample_rate // gcd,
        ).astype(np.float32, copy=False)
    except Exception:
        pass

    duration = len(audio) / float(source_sample_rate)
    target_count = max(1, int(round(duration * target_sample_rate)))
    source_times = np.arange(len(audio), dtype=np.float64) / float(source_sample_rate)
    target_times = np.arange(target_count, dtype=np.float64) / float(target_sample_rate)
    return np.interp(target_times, source_times, audio).astype(np.float32)


def clamp_and_merge_speech_segments(
    segments: Sequence[SpeechSegment],
    duration_seconds: float,
    tolerance_seconds: float,
    min_duration_seconds: float = 0.0,
    merge_gap_seconds: float = 0.02,
) -> list[SpeechSegment]:
    if duration_seconds <= 0.0:
        return []

    normalized = []
    for segment in segments:
        start = max(0.0, float(segment.start_time) - tolerance_seconds)
        end = min(duration_seconds, float(segment.end_time) + tolerance_seconds)
        if end - start >= min_duration_seconds:
            normalized.append(SpeechSegment(start, end))

    normalized.sort(key=lambda item: item.start_time)
    merged: list[SpeechSegment] = []
    for segment in normalized:
        if not merged or segment.start_time > merged[-1].end_time + merge_gap_seconds:
            merged.append(segment)
            continue

        previous = merged[-1]
        merged[-1] = SpeechSegment(
            previous.start_time,
            max(previous.end_time, segment.end_time),
        )

    return merged


def speech_segments_to_frame_mask(
    segments: Sequence[SpeechSegment],
    frame_count: int,
    frame_duration_seconds: float,
) -> np.ndarray:
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if frame_count == 0:
        return np.zeros(0, dtype=bool)
    if frame_duration_seconds <= 0.0:
        raise ValueError("frame_duration_seconds must be positive")

    mask = np.zeros(frame_count, dtype=bool)
    for segment in segments:
        start_frame = max(0, int(math.floor(segment.start_time / frame_duration_seconds)))
        end_frame = min(
            frame_count,
            int(math.ceil(segment.end_time / frame_duration_seconds)),
        )
        if end_frame > start_frame:
            mask[start_frame:end_frame] = True
    return mask


def _active_frames_to_segments(
    active: Sequence[bool],
    hop_duration_seconds: float,
    frame_duration_seconds: float,
) -> list[SpeechSegment]:
    segments: list[SpeechSegment] = []
    start_index = None
    for index, is_active in enumerate(active):
        if is_active and start_index is None:
            start_index = index
        elif not is_active and start_index is not None:
            segments.append(
                SpeechSegment(
                    start_index * hop_duration_seconds,
                    index * hop_duration_seconds + frame_duration_seconds,
                )
            )
            start_index = None

    if start_index is not None:
        segments.append(
            SpeechSegment(
                start_index * hop_duration_seconds,
                len(active) * hop_duration_seconds + frame_duration_seconds,
            )
        )
    return segments
