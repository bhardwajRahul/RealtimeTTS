from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Optional

from .._audio_backend import pyaudio

from .audio import (
    audio_bytes_to_mono_float32,
    compressed_audio_duration_seconds,
    pcm_chunk_duration_seconds,
)


@dataclass
class CapturedAudioSegment:
    text: str
    start_time: float
    audio_bytes: bytes
    stream_format: int
    channels: int
    sample_rate: int

    def to_mono_float32(self):
        return audio_bytes_to_mono_float32(
            self.audio_bytes,
            self.stream_format,
            self.channels,
            self.sample_rate,
        )


class CapturingAudioQueue:
    """
    Queue proxy that tees synthesized audio into one active sentence capture.
    """

    def __init__(
        self,
        wrapped_queue: queue.Queue,
        stream_format: int,
        channels: int,
        sample_rate: int,
    ):
        self._queue = wrapped_queue
        self.stream_format = stream_format
        self.channels = channels
        self.sample_rate = sample_rate
        self.total_duration = 0.0
        self._lock = threading.Lock()
        self._active_text: Optional[str] = None
        self._active_chunks: list[bytes] = []
        self._active_start_time = 0.0

    def start_capture(self, text: str) -> bool:
        with self._lock:
            if self._active_text is not None:
                return False
            self._active_text = text
            self._active_chunks = []
            self._active_start_time = self.total_duration
            return True

    def reset_tracking(self) -> None:
        with self._lock:
            self.total_duration = 0.0
            self._active_text = None
            self._active_chunks = []
            self._active_start_time = 0.0

    def finish_capture(self) -> Optional[CapturedAudioSegment]:
        with self._lock:
            if self._active_text is None:
                return None

            text = self._active_text
            start_time = self._active_start_time
            audio_bytes = b"".join(self._active_chunks)
            self._active_text = None
            self._active_chunks = []
            self._active_start_time = 0.0

            if self.stream_format == pyaudio.paCustomFormat and audio_bytes:
                try:
                    self.total_duration += compressed_audio_duration_seconds(audio_bytes)
                except Exception:
                    pass

            return CapturedAudioSegment(
                text=text,
                start_time=start_time,
                audio_bytes=audio_bytes,
                stream_format=self.stream_format,
                channels=self.channels,
                sample_rate=self.sample_rate,
            )

    def put(self, item: Any, *args, **kwargs):
        item_bytes = self._coerce_bytes(item)
        with self._lock:
            if item_bytes is not None and self._active_text is not None:
                self._active_chunks.append(item_bytes)

            duration = None
            if item_bytes is not None and self.stream_format != pyaudio.paCustomFormat:
                duration = pcm_chunk_duration_seconds(
                    item_bytes,
                    self.stream_format,
                    self.channels,
                    self.sample_rate,
                )
            if duration:
                self.total_duration += duration

        return self._queue.put(item, *args, **kwargs)

    def get(self, *args, **kwargs):
        return self._queue.get(*args, **kwargs)

    def get_nowait(self):
        return self._queue.get_nowait()

    def empty(self):
        return self._queue.empty()

    def qsize(self):
        return self._queue.qsize()

    def task_done(self):
        return self._queue.task_done()

    def join(self):
        return self._queue.join()

    @staticmethod
    def _coerce_bytes(item: Any) -> Optional[bytes]:
        if isinstance(item, bytes):
            return item
        if isinstance(item, (bytearray, memoryview)):
            return bytes(item)
        return None
