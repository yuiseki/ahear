# ahear

Agentic Hearing

`ahear` is the hearing module of the `a*` stack.

Current status:

- `python/src/ahear/whisper_listener.py`
  - extracted from the legacy `tmp/` runtime
  - keeps the current `parec + whisper-server` compatibility contract
- `python/src/ahear/moonshine_listener.py`
  - extracted from the legacy `tmp/` runtime
  - keeps the current `ffmpeg + moonshine` compatibility contract
- `python/src/ahear/speaker_auth.py`
  - extracted from the legacy `tmp/` runtime
  - keeps the current speaker verification primitive while biometric policy stays outside `ahear`
- `python/src/ahear/collect_guest_voice.py`
  - extracted from the legacy `tmp/` runtime
  - collects non-owner voice samples with energy VAD into a persistent local dataset
- `python/src/ahear/models/master_voiceprint.npy`
  - local speaker master cache for the current compatibility runtime
  - intentionally gitignored, following the same pattern as `repos/asee/python/src/asee/models`

The canonical listener entrypoints are now `python/src/ahear/whisper_listener.py` and
`python/src/ahear/moonshine_listener.py` — launched via `yuiclaw voice-command operator`.

For guest-data collection, the canonical compatibility entrypoint is now:

- `python/src/ahear/collect_guest_voice.py`
  - default output: `/home/yuiseki/Workspaces/private/datasets/voices/others`
  - default source keywords: `dji,razer`

This repo is still expected to evolve toward a cleaner long-term surface, but
the current priority is to preserve the working runtime contract while migrating
code out of `tmp/`.
