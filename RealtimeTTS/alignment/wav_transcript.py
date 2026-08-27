from __future__ import annotations

import os
from typing import Sequence

from .audio import wav_file_to_mono_float32
from .base import CharacterAligner, CharacterAlignment


class WavTranscriptCharacterAligner:
    """
    Reusable adapter for forced-aligning a transcript against a WAV file.
    """

    def __init__(self, aligner: CharacterAligner):
        self.aligner = aligner

    def align_file(
        self,
        wav_path: str | os.PathLike,
        transcript: str,
        channel: int | None = None,
    ) -> Sequence[CharacterAlignment]:
        audio, sample_rate = wav_file_to_mono_float32(wav_path, channel=channel)
        return self.aligner.align_characters(transcript, audio, sample_rate)


def align_wav_transcript_characters(
    wav_path: str | os.PathLike,
    transcript: str,
    channel: int | None = None,
    aligner: CharacterAligner | None = None,
    backend: str = "mms_fa",
    device: str | None = None,
    use_vad: bool = True,
    vad_tolerance_seconds: float = 0.05,
    vad_supplemental_after_coarse_start: bool = True,
    vad_supplemental_start_margin_seconds: float = 0.2,
) -> Sequence[CharacterAlignment]:
    if aligner is None:
        if backend == "mms_fa":
            from .torchaudio_fa import TorchaudioMMSFACharacterAligner

            aligner = TorchaudioMMSFACharacterAligner(device=device)
            return aligner.align_wav_characters(
                wav_path,
                transcript,
                channel=channel,
            )
        if backend == "omniasr":
            from .omniasr import OmniASRCTCAligner

            aligner = OmniASRCTCAligner(
                device=device,
                use_vad=use_vad,
                vad_tolerance_seconds=vad_tolerance_seconds,
                vad_supplemental_after_coarse_start=(
                    vad_supplemental_after_coarse_start
                ),
                vad_supplemental_start_margin_seconds=(
                    vad_supplemental_start_margin_seconds
                ),
            )
            return aligner.align_wav_characters(
                wav_path,
                transcript,
                channel=channel,
                use_vad=use_vad,
                vad_tolerance_seconds=vad_tolerance_seconds,
            )
        raise ValueError("backend must be 'mms_fa' or 'omniasr'")

    return WavTranscriptCharacterAligner(aligner).align_file(
        wav_path,
        transcript,
        channel=channel,
    )
