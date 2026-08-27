"""Benchmark warmed Breeze TTS 2 queue and audible-onset latency.

The first request and warmups are kept separate from reported warm runs.  The
tool measures the RealtimeTTS queue boundary; sound-device startup is outside
its scope.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import queue
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from RealtimeTTS import BreezeTTSEngine, BreezeTTSVoice


class RecordingQueue(queue.Queue):
    def __init__(self) -> None:
        super().__init__()
        self.arrivals: list[tuple[int, bytes]] = []

    def put(self, item, block=True, timeout=None):
        self.arrivals.append((time.perf_counter_ns(), item))
        return super().put(item, block=block, timeout=timeout)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(runs)}
    for field in ("queue_ttfa_ms", "audible_onset_ms", "wall_ms", "rtf"):
        values = [float(run[field]) for run in runs if run.get(field) is not None]
        if values:
            result[field] = {
                "median": statistics.median(values),
                "mean": statistics.fmean(values),
                "p95": _percentile(values, 0.95),
            }
    for field in ("cuda_peak_allocated_mib", "cuda_peak_reserved_mib"):
        values = [
            float(run["engine_metrics"][field])
            for run in runs
            if run["engine_metrics"].get(field) is not None
        ]
        if values:
            result[field] = {"max": max(values), "median": statistics.median(values)}
    return result


def _first_audible_ms(
    arrivals: list[tuple[int, bytes]],
    started_ns: int,
    sample_rate: int,
    threshold: int,
) -> float | None:
    playback_cursor_ns: int | None = None
    for arrival_ns, pcm in arrivals:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
        chunk_start_ns = max(arrival_ns, playback_cursor_ns or arrival_ns)
        audible = np.flatnonzero(np.abs(samples) >= threshold)
        if audible.size:
            return (
                (chunk_start_ns - started_ns) / 1_000_000
                + int(audible[0]) / sample_rate * 1000.0
            )
        playback_cursor_ns = chunk_start_ns + int(
            samples.size / sample_rate * 1_000_000_000
        )
    return None


def _run_once(
    engine: BreezeTTSEngine,
    text: str,
    index: int,
    audible_threshold: int,
    audio_output: Path | None = None,
) -> dict[str, Any]:
    recording_queue = RecordingQueue()
    engine.queue = recording_queue
    started_ns = time.perf_counter_ns()
    success = engine.synthesize(text)
    ended_ns = time.perf_counter_ns()
    if not success:
        raise RuntimeError(
            f"Breeze synthesis failed: {engine.last_synthesis_metrics.get('error')}"
        )

    pcm = b"".join(chunk for _, chunk in recording_queue.arrivals)
    pcm_bytes = len(pcm)
    audio_seconds = pcm_bytes / 2.0 / engine.sampling_rate
    if audio_output is not None:
        audio_output.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(audio_output), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(engine.sampling_rate)
            wav.writeframes(pcm)
    queue_ttfa_ms = (
        (recording_queue.arrivals[0][0] - started_ns) / 1_000_000
        if recording_queue.arrivals
        else None
    )
    wall_ms = (ended_ns - started_ns) / 1_000_000
    return {
        "index": index,
        "queue_ttfa_ms": queue_ttfa_ms,
        "audible_onset_ms": _first_audible_ms(
            recording_queue.arrivals,
            started_ns,
            engine.sampling_rate,
            audible_threshold,
        ),
        "wall_ms": wall_ms,
        "audio_seconds": audio_seconds,
        "rtf": wall_ms / 1000.0 / audio_seconds if audio_seconds else None,
        "chunks": len(recording_queue.arrivals),
        "pcm_bytes": pcm_bytes,
        "audio_output": str(audio_output) if audio_output is not None else None,
        "engine_metrics": dict(engine.last_synthesis_metrics),
    }


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "realtimetts",
        "torch",
        "transformers",
        "bitsandbytes",
        "qwen-tts",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "source-checkout"
    return versions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breeze-root", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--model-id", default="BreezeBlue/Breeze-TTS-2")
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--quantization", choices=("none", "int8", "nf4"), default="nf4"
    )
    parser.add_argument("--int8-threshold", type=float, default=6.0)
    parser.add_argument("--quantization-skip-module", action="append", default=[])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--audible-threshold", type=int, default=256)
    parser.add_argument(
        "--instruction",
        default="A calm, warm narrator with clear articulation.",
    )
    parser.add_argument(
        "--text",
        default="This is a short warmed latency benchmark for Breeze TTS 2.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audio-output", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.runs <= 0:
        parser.error("--warmups must be >= 0 and --runs must be > 0")
    return args


def main() -> int:
    args = _parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch build is required.")
    device = torch.device(args.device)
    free_before, total = torch.cuda.mem_get_info(device)
    environment = {
        "platform": platform.platform(),
        "python": sys.version,
        "packages": _package_versions(),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "cuda_runtime": torch.version.cuda,
        "cuda_free_before_load_mib": free_before / 1024**2,
        "cuda_total_mib": total / 1024**2,
    }

    load_started = time.perf_counter()
    engine = BreezeTTSEngine(
        breeze_root=args.breeze_root,
        model_path=args.model_path,
        model_id=args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
        llm_int8_threshold=args.int8_threshold,
        quantization_skip_modules=args.quantization_skip_module,
        voice=BreezeTTSVoice(
            instruction=args.instruction,
            cfg_scale=args.cfg_scale,
        ),
        seed=args.seed,
        fast_all=False,
    )
    load_wall_ms = (time.perf_counter() - load_started) * 1000.0
    try:
        first_request = _run_once(engine, args.text, 0, args.audible_threshold)
        warmups = [
            _run_once(engine, args.text, index + 1, args.audible_threshold)
            for index in range(args.warmups)
        ]
        runs = [
            _run_once(
                engine,
                args.text,
                index + 1,
                args.audible_threshold,
                args.audio_output if index == 0 else None,
            )
            for index in range(args.runs)
        ]
        result = {
            "environment": environment,
            "configuration": {
                "quantization": args.quantization,
                "int8_threshold": args.int8_threshold,
                "quantization_skip_modules": args.quantization_skip_module,
                "dtype": args.dtype,
                "warmups_excluded": args.warmups,
                "runs": args.runs,
                "text": args.text,
                "instruction": args.instruction,
                "cfg_scale": args.cfg_scale,
                "audible_threshold": args.audible_threshold,
            },
            "load_wall_ms": load_wall_ms,
            "load_metrics": engine.load_metrics,
            "first_request": first_request,
            "warmups": warmups,
            "measured_runs": runs,
            "summary": _summarize(runs),
        }
    finally:
        engine.shutdown()

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
