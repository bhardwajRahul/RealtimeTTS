# Changelog

## 0.8.2 - 2026-08-28

### Fixed

- Smooth long Qwen fragment boundaries by retaining and fading the final 15 ms
  after the first second of audio without delaying startup or short fragments.
- Add configurable 150 ms mid-sentence and 300 ms sentence-ending pauses to
  Qwen WebSocket text streaming.
- Reduce the Qwen silence-trim onset fade from 20 ms to 15 ms.

### Release process

- Harden deployed-runtime attestations, service verification, path
  normalization, and remote release-ref validation.

## 0.8.1 - 2026-08-27

### Fixed

- Warm every configured Qwen language-route voice before the server becomes
  ready, with the route's actual synthesis language and without duplicate work.
- Keep compatible generated Qwen voice caches across Python binding package
  version changes; native ABI/version and model identities remain part of the
  cache contract.

### Release process

- Require an exact deployed-wheel attestation before publication and block
  releases when any linked worktree contains uncommitted changes.

## 0.8.0 - 2026-08-27

### Added

- `HiggsEngine`, a small raw-PCM streaming client for separately operated
  SGLang-Omni Higgs Audio v3 servers, with response-header validation and
  fallback-safe failures.
- Optional forced word and character alignment APIs, sentence-audio capture,
  MMS/torchaudio and omniASR adapters, VAD helpers, and `on_word` integration.
- A `language` selector for the CPU PocketTTS engine.
- A bounded Kokoro blended-voice cache and explicit native-timing capability
  flags for Azure, Cartesia, and Kokoro.
- Acknowledged native Qwen pause/resume controls for the WebSocket server.
- Opt-in fragment lookahead through the public `stream2sentence` 1.0.4 release.

### Changed

- Require Python 3.10 through 3.14 for the core package.

### Fixed

- Treat empty Higgs and Kokoro synthesis results as failures instead
  of successful silent output.
- Prevent late nonblocking alignment results from leaking into a later playback.
- Make immediate playback stop finish cleanup deterministically after the
  playback worker exits, with a bounded watchdog for blocked audio backends.
- Preserve explicit Kokoro language overrides, reject mixed-language or invalid
  blend formulas, and release cached pipelines and queues during shutdown.
- Initialize PocketTTS GPU stateful-module names and restore nested cached state
  keys correctly.

## 0.7.4

### Added

- Native `QwenEngine` integration through the ABI-4 `qwentts.cpp` binding.
- Installable headless Qwen server via `realtimetts[qwen-server]` and the
  `realtimetts-qwen-server` console command.
- Persistent Qwen voice registration, incremental text streaming, language
  routing, health/readiness reporting, and production request controls.
- `InflectEngine` with pinned PyTorch and ONNX model revisions.
- A shared package version exposed as `RealtimeTTS.__version__`.

### Changed

- Replaced `FasterQwenEngine` with the native `QwenEngine` implementation.
- Made Stanza opt-in and selected the NLTK plus rule-based tokenizer by default.
- Kept the server extra headless while engine playback extras continue to use
  PyAudio/PortAudio.
- Selected `realtimetts-qwen-native` 0.1.0 for the validated Windows and Linux
  CUDA 12.8 Qwen wheels. Unsupported platforms are not release targets.

### Fixed

- Preserved host logging configuration while loading the Inflect runtime.
- Hardened streaming silence trimming and short-utterance startup behavior.
- Added deterministic cancellation, bounded concurrency, and graceful shutdown
  behavior to the Qwen server.
- Fixed WebSocket cancellation after the `end` event so active PCM synthesis
  stops promptly and emits the terminal `cancelled` event.
