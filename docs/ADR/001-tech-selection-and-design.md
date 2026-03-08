# ADR 001: ahear — Technology Selection and Design

## Status

Accepted

## Context

`ahear` is the **Ears** of the AI agent body (see [yuiclaw ADR 002](../../yuiclaw/docs/ADR/002-sensorimotor-unix-philosophy.md)).

**Single Responsibility:** Capture microphone audio, detect speech, transcribe to text, output to stdout / acomm.

### PoC Reference

`/home/yuiseki/Workspaces/tmp/whispercpp-listen/`

The PoC (`listen_only_whisper_server.py`, `listen_only_moonshine_server.py`,
`voice_command_loop.py`) demonstrates the full pipeline but violates the
single-responsibility principle by also:
- Parsing voice commands
- Calling VOICEVOX TTS (→ belongs in `asay`)
- Controlling VacuumTube (→ belongs in `adesk`)

`ahear` extracts only the **hear** portion: mic → VAD → STT → text.

As of 2026-03-08, the first live migration slice is already implemented in
Python:

- `python/src/ahear/whisper_listener.py`
- `python/src/ahear/moonshine_listener.py`
- `python/src/ahear/speaker_auth.py`

The legacy files in `tmp/whispercpp-listen/` are now thin compatibility
wrappers that delegate to these modules.

## Decision

### Language & Runtime

Long term: **Rust** remains the target implementation language because it fits
the `a*` ecosystem (`acore`, `abeat`, `amem`, `acomm`) and is a better home for
the final long-running audio pipeline.

Current migration slice: **Python** compatibility modules are accepted as the
first extraction step because they let us move working runtime code out of
`tmp/whispercpp-listen` without changing behaviour first.

### Multi-Device Design Principle

**「1プロセス = 1デバイス」** を基本とする。複数マイクを使用する場合は、複数の `ahear` プロセスを独立して起動し、それぞれが `acomm` に publish する。プロセス間の調整は `acomm` と `abeat` が担い、`ahear` 自身は他インスタンスの存在を関知しない。

```sh
# abeat が管理する起動例
ahear --source "DJI MIC MINI" --publish &      # ワイヤレスラベリアマイク
ahear --source "Razer Seiren Mini" --publish & # デスクトップマイク
```

各インスタンスの出力 JSON には `source` フィールドを含め、`acore` 側でどのデバイスからの発話かを区別できるようにする。

現在接続中のマイク:
- `DJI MIC MINI` — ワイヤレスラベリアマイク
- `Razer Seiren Mini` — USB コンデンサーマイク
- `HD Pro Webcam C920` (×2) — 内蔵マイク（必要に応じて使用）

### Audio Capture

Whisper compatibility path uses **`parec`** (PulseAudio record,
PipeWire-compatible) for audio capture:
- Available on the target system (Ubuntu + PipeWire PulseAudio compat layer)
- Outputs raw PCM to stdout — easy to pipe
- Source device selectable via `--source` flag (supports DJI MIC MINI, Razer, etc.)

```sh
parec --format=s16le --rate=16000 --channels=1 --device=<source>
```

### Voice Activity Detection (VAD)

Energy-based VAD implemented in Rust:
- RMS threshold: configurable via `--vad-threshold` (default: 300)
- Minimum speech chunk: configurable via `--min-duration` (default: 0.5s)
- Silence timeout: configurable via `--silence-timeout` (default: 1.0s)
- Buffers audio until silence detected, then ships chunk to whisper

Rationale: The PoC's Python energy-based VAD (`sum(abs(x) for x in chunk)`) is simple and effective. No external VAD library dependency needed for initial implementation.

### Speech Recognition

The extracted compatibility layer currently supports two backends:

1. `whisper_listener`
   - **whisper.cpp HTTP server** (`whisper-server`)
2. `moonshine_listener`
   - in-process `moonshine_voice` transcription

For the whisper path:
- Already deployed on the target system
- REST API: `POST /inference` with WAV body → JSON response
- Model selection: `small` (fast) or `medium` (accurate), runtime-swappable
- Language: Japanese (`--language ja`)

`ahear` does **not** embed whisper.cpp. It is a client to the already-running
server. `abeat` is responsible for ensuring the server is running.

### Output Format

**stdout** — one JSON line per recognized utterance:

```json
{"source": "DJI MIC MINI", "text": "今日の天気はどうですか", "timestamp": "2026-02-27T10:15:30+09:00", "duration_ms": 1240}
```

`source` フィールドにより、複数インスタンスの出力を `acomm` や `acore` 側でデバイスごとに識別できる。

Newline-delimited JSON (NDJSON) enables direct piping:

```sh
ahear --source "DJI MIC MINI" | acore
ahear --source "Razer Seiren Mini" | jq '.text' | asay
# 複数インスタンスの出力を acomm 経由でマージ
ahear --source "DJI MIC MINI" --publish &
ahear --source "Razer Seiren Mini" --publish &
```

### acomm Integration

When `--publish` flag is set, additionally publish each utterance to `acomm` as a `RecognizedSpeech` event:

```json
{"type": "RecognizedSpeech", "source": "DJI MIC MINI", "text": "...", "timestamp": "..."}
```

### CLI Interface

```
ahear [OPTIONS]

Current extracted compatibility commands are still driven by the legacy CLI
surface from `tmp/whispercpp-listen`:

- `listen_only_whisper_server.py`
- `listen_only_moonshine_server.py`

The stable `ahear` CLI should be introduced after the migration slices stop
depending on the PoC argument surface.
```

複数マイクを同時監視する場合は、別プロセスで起動する:

```sh
ahear --source "DJI MIC MINI" --publish &
ahear --source "Razer Seiren Mini" --publish &
```

### Daemon Management

`ahear` itself is **not** a daemon manager. Long-running operation is handled by:
- `tmux` session (manual)
- `abeat` job definition (automated)

The PoC's `tmux_listen_only.sh` logic should eventually migrate to an `abeat`
job definition, but it currently still orchestrates the compatibility wrappers.

## Consequences

- `ahear` already owns the extracted STT backend implementation, even though the
  operator-facing entrypoint is still the legacy wrapper in `tmp/whispercpp-listen`
- `ahear` now also owns the speaker verification primitive, while lock/unlock
  policy still stays outside the repo
- **「1プロセス = 1デバイス」** により、インスタンスを増やすだけで監視対象マイクを拡張できる
- `source` フィールドにより複数インスタンスの出力を `acore` 側で識別可能
- Voice command parsing is fully removed from `ahear` (moved to `acore`)
- VOICEVOX calls are fully removed from `ahear` (moved to `asay`)
- The PoC's daemon shell script is replaced by an `abeat` job
- whisper.cpp server lifecycle is managed externally (by `abeat` or the user)

## Implementation Plan

### MUST（優先実装）

1. Keep the extracted Python compatibility listeners green with contract tests
2. Move remaining STT orchestration out of `tmp/whispercpp-listen`
3. Introduce a stable `ahear` operator-facing CLI separate from PoC wrappers
4. Decide whether the long-term production runtime should stay Python or be
   rewritten in Rust
