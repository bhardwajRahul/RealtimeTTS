# Breeze TTS 2

> **Important:** the inference source is Apache-2.0, but the model weights,
> quantized or other derivative models, and self-hosted outputs are restricted
> to research and non-commercial use. Commercial use requires written upstream
> authorization. You are responsible for consent and rights for reference
> voices, input material, and generated audio.

`BreezeTTSEngine` wraps the official Breeze TTS 2 PyTorch streaming runtime.
It supports English and Chinese voice design, reference voice cloning, and
reference-guided voice direction. Audio is queued as mono 24 kHz signed 16-bit
PCM.

## Status and hardware reality

The official project documents Linux, Python 3.10+, CUDA, approximately
7.7 GiB of eager GPU memory, and a 12 GB minimum recommendation. Its
`--fast-all` configuration is documented at approximately 14.4 GiB with a
24 GB recommendation. The advertised sub-40 ms TTFA is specifically a warmed
H100 fast-path result, not a Windows or end-to-end playback guarantee.

The original checkpoint contains about 6.49 GiB of main BF16 weights plus a
roughly 0.64 GiB bundled Mimi/Qwen audio tokenizer. On an 8 GB RTX 2080-class
card, use `quantization="int8"` or `quantization="nf4"`; the unquantized path
does not have safe runtime headroom. Quantized execution is experimental and
all Breeze CUDA-graph fast flags must remain disabled.

A structural dry run against the official config and tensor index finds about
2.495 billion weights in quantizable linear layers. The calculated main-model
storage floors are therefore about 4.16 GiB for INT8 and 3.00 GiB for NF4,
before the 0.64 GiB codec, quantization metadata, KV cache, activations, and
CUDA workspaces. These are sizing estimates, not measured peak VRAM.

## License

The upstream inference source is Apache-2.0. The Breeze TTS 2 weights,
derivative models (including quantized copies), and self-hosted outputs use the
BreezeBlue Research and Non-Commercial License. Commercial use requires the
upstream licensor's written authorization.

## Installation

Use Python 3.10-3.13. The official upstream target is Linux with CUDA; the
Windows INT8/NF4 path is experimental and not a supported release target.
Install the CUDA build before the RealtimeTTS extra; the
plain Windows PyPI wheel for the upstream `torch==2.9.1` requirement is CPU-only.

```powershell
python -m venv .venv-breeze
.\.venv-breeze\Scripts\python.exe -m pip install `
  torch==2.9.1 torchaudio==2.9.1 `
  --index-url https://download.pytorch.org/whl/cu128
.\.venv-breeze\Scripts\python.exe -m pip install -e ".[breeze]"
git clone https://github.com/breezeblue-ai/breeze-tts D:\path\to\breeze-tts
git -C D:\path\to\breeze-tts checkout ca632ce6c4d05f7985da4eab29b1a5d445b43f7b
```

The adapter is pinned to inference-code revision
`ca632ce6c4d05f7985da4eab29b1a5d445b43f7b` and defaults to model revision
`c1c8ca18b70b30822735633991d9ebf4898e47d4`. These pins matter because the
adapter imports internal upstream modules. Pass a different model `revision`
only after retesting the integration. The Breeze stack is intentionally not
included in `realtimetts[all]`; install `realtimetts[breeze]` explicitly.

CI exercises the public adapter, PCM conversion, error behavior, and
quantization configuration with a deterministic fake backend. Treat a real
supported Linux/CUDA synthesis smoke and benchmark as an additional acceptance
gate before promoting Breeze from experimental use.

On Windows, install the external SoX executable and make sure `sox.exe` is on
`PATH`. The Python `sox` package alone is not sufficient.

The engine downloads `BreezeBlue/Breeze-TTS-2` on first construction unless
`model_path` points at an existing snapshot. That download is approximately
7.15 GiB even when weights are quantized during loading; bitsandbytes reduces
VRAM use, not the original download size. The `int8` and `nf4` modes quantize
the original checkpoint while loading and do not require a separately
published quantized checkpoint.

## INT8 example

```python
from RealtimeTTS import BreezeTTSEngine, BreezeTTSVoice, TextToAudioStream


if __name__ == "__main__":
    voice = BreezeTTSVoice(
        name="calm",
        instruction="A calm, warm narrator with clear articulation.",
        cfg_scale=1.0,
    )
    engine = BreezeTTSEngine(
        breeze_root=r"D:\path\to\breeze-tts",
        voice=voice,
        quantization="int8",  # aliases: "q8", "8-bit"
        dtype="float16",      # appropriate for Turing / RTX 20-series
        fast_all=False,
    )
    stream = TextToAudioStream(engine)
    stream.feed("This is a quantized Breeze TTS 2 test.").play()
    print(engine.load_metrics)
    print(engine.last_synthesis_metrics)
    engine.shutdown()
```

Use `quantization="nf4"` (aliases `q4` and `4-bit`) when INT8 still leaves too
little headroom. The engine keeps `lm_head` and `codebooks_head` unquantized
because the official eager streaming runtime explicitly casts those heads to
FP32.

## Reference-guided voice

Reference audio and its exact transcript must be supplied together:

```python
voice = BreezeTTSVoice(
    name="reference",
    ref_audio_path=r"D:\voices\reference.wav",
    ref_text="This is the exact transcript of the reference audio.",
    instruction="Keep the voice identity and speak slowly.",
    cfg_scale=4.0,
)
```

`last_synthesis_metrics` reports first queued chunk latency, total generation
time, generated audio duration, RTF, and CUDA allocator peaks. This is model to
RealtimeTTS-queue TTFA; speaker-device startup and audible voice onset are
separate measurements.

For repeatable cold/warm measurements, use the included benchmark. It keeps the
first request and two warmups out of the five-run summary and reports both the
first queued PCM chunk and the first non-silent sample:

```powershell
python tools\benchmark_breeze_engine.py `
  --breeze-root D:\path\to\breeze-tts `
  --quantization nf4 `
  --warmups 2 --runs 5 `
  --output D:\Temp\breeze-nf4-benchmark.json
```
