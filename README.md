# ahear

Agentic Hearing

`ahear` is the hearing module of the `a*` stack.

Current status:

- `python/src/ahear/whisper_listener.py`
  - extracted from `tmp/whispercpp-listen/listen_only_whisper_server.py`
  - keeps the current `parec + whisper-server` compatibility contract
- `python/src/ahear/moonshine_listener.py`
  - extracted from `tmp/whispercpp-listen/listen_only_moonshine_server.py`
  - keeps the current `ffmpeg + moonshine` compatibility contract
- `python/src/ahear/speaker_auth.py`
  - extracted from `tmp/whispercpp-listen/voice_command_loop.py`
  - keeps the current speaker verification primitive while biometric policy stays outside `ahear`
- `python/src/ahear/models/master_voiceprint.npy`
  - local speaker master cache for the current compatibility runtime
  - intentionally gitignored, following the same pattern as `repos/asee/python/src/asee/models`

`tmp/whispercpp-listen/listen_only_whisper_server.py` and
`tmp/whispercpp-listen/listen_only_moonshine_server.py` now exist as thin
compatibility wrappers that delegate to `ahear`.

This repo is still expected to evolve toward a cleaner long-term surface, but
the current priority is to preserve the working runtime contract while migrating
code out of `tmp/`.
