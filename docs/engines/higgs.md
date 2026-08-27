# Higgs Audio v3 Engine

`HiggsEngine` connects RealtimeTTS to a separately running SGLang-Omni
Higgs Audio v3 server. It uses the OpenAI-compatible
`POST /v1/audio/speech` endpoint and requests the current raw streaming
contract: `stream=true` with `response_format="pcm"`.

## Installation

Install the small RealtimeTTS-side dependency set:

```bash
pip install "realtimetts[higgs]"
```

Install, configure, and run
[SGLang-Omni's Higgs server](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/higgs_tts.md)
separately. Its model weights, CUDA runtime, and server are not distributed by
RealtimeTTS. Review their current licenses and deployment documentation before
use.

## Usage

```python
from RealtimeTTS import HiggsEngine, HiggsVoice, TextToAudioStream


engine = HiggsEngine(
    api_url="http://127.0.0.1:8000/v1/audio/speech",
    model="bosonai/higgs-audio-v3-tts-4b",
    voice=HiggsVoice(
        name="default",
        prefix="<|emotion:calm|><|prosody:expressive_low|>",
    ),
)
stream = TextToAudioStream(engine)
stream.feed("Hello from the Higgs server.").play()
engine.shutdown()
```

Set `HIGGS_TTS_API_URL` instead of passing `api_url` when preferred. Use
the `headers` argument for authentication when the server is not a trusted
local process. Do not expose an unauthenticated synthesis server to an
untrusted network.

The `model` field is optional when the server's loaded model is its default.
`initial_codec_chunk_frames` is also optional; SGLang-Omni currently gives
Higgs a continuity-safe default of 20. Values including zero can be used for
explicit TTFA/continuity experiments. Model-specific fields such as
`references`, `language`, or `repetition_penalty` can be supplied through
`extra_payload`.

## Streaming contract

The engine requires raw signed 16-bit little-endian mono PCM. HTTP EOF ends the
stream; there are no SSE events, base64 envelopes, or terminal sentinels.
`Content-Type`, `X-Sample-Rate`, `X-Channels`, and `X-Bit-Depth` are
validated when the server supplies them. The configured default is 24 kHz,
matching Higgs Audio v3.

Network chunks do not have to end on an audio-frame boundary: the engine keeps
a one-byte remainder until the next chunk. A final partial frame, incompatible
metadata, HTTP error, or successful response containing no audio is treated as
a synthesis failure so a RealtimeTTS fallback engine can take over.

Generated voices and audio remain subject to the Higgs model/server license,
the rights attached to any reference material, and applicable consent and
impersonation laws.
