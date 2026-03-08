#!/usr/bin/env python3
"""Listen-only microphone transcription using ffmpeg + simple energy VAD + moonshine.

Drop-in replacement for listen_only_whisper_server.py.  The public API
(ListenLoop class, module-level constants and helper functions) is intentionally
identical so that voice_command_loop.py can switch back-ends by changing a
single import line:

    # whisper backend (original)
    import listen_only_whisper_server as base

    # moonshine backend (this file)
    import listen_only_moonshine_server as base

Differences from the whisper variant
-------------------------------------
- Audio capture  : ffmpeg -f pulse  (instead of parec)
- STT back-end   : moonshine_voice.Transcriber  (instead of HTTP → whisper-server)
- --server arg   : not supported; ignored with a warning when received
- --language arg : not supported; moonshine model is language-specific (ja)
- --stt-prompt   : moonshine has no prompt injection; accepted but silently ignored
- New --model-size {tiny,base} : selects moonshine-tiny-ja or moonshine-base-ja
"""

from __future__ import annotations

import argparse
import array
import collections
import datetime as dt
import math
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


# ── Constants (identical to listen_only_whisper_server.py) ───────────────────

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # s16le
BYTES_PER_SAMPLE = CHANNELS * SAMPLE_WIDTH


# ── Helpers identical to listen_only_whisper_server.py ───────────────────────

def now_ts() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def pcm_rms(chunk: bytes) -> float:
    if not chunk:
        return 0.0
    n = len(chunk) // 2
    if n <= 0:
        return 0.0
    samples = struct.unpack("<" + "h" * n, chunk[: n * 2])
    if not samples:
        return 0.0
    mean_sq = sum(s * s for s in samples) / len(samples)
    return math.sqrt(mean_sq) / 32768.0


def wav_bytes_from_pcm(raw_pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Minimal RIFF/WAV writer for PCM16 mono — unchanged from whisper variant."""
    data_size = len(raw_pcm)
    byte_rate = sample_rate * CHANNELS * SAMPLE_WIDTH
    block_align = CHANNELS * SAMPLE_WIDTH
    riff_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        CHANNELS,
        sample_rate,
        byte_rate,
        block_align,
        8 * SAMPLE_WIDTH,
        b"data",
        data_size,
    )
    return header + raw_pcm


def list_sources() -> list[str]:
    cp = subprocess.run(
        ["pactl", "list", "short", "sources"],
        check=True, text=True, capture_output=True,
    )
    names: list[str] = []
    for line in cp.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            names.append(parts[1])
    return names


def detect_source(user_source: str | None) -> str:
    if user_source:
        return user_source
    names = list_sources()
    for n in names:
        if "DJI_MIC_MINI" in n:
            return n
    for n in names:
        if "Razer_Seiren_Mini" in n:
            return n
    for n in names:
        if n.startswith("alsa_input.usb-") and ".monitor" not in n and "C920" not in n:
            return n
    for n in names:
        if n.startswith("alsa_input.") and ".monitor" not in n:
            return n
    cp = subprocess.run(["pactl", "info"], check=True, text=True, capture_output=True)
    for line in cp.stdout.splitlines():
        if line.startswith("Default Source: "):
            return line.split(": ", 1)[1].strip()
    raise RuntimeError("No usable PulseAudio/PipeWire source found")


# ── Moonshine transcriber singleton ──────────────────────────────────────────
#
# Kept at module level so that voice_command_loop.py can call the free function
# transcribe_with_server() without holding a reference to the Transcriber object.

_transcriber: Any | None = None
_transcriber_lock = threading.Lock()
_stt_prompt_warned = False


# ── Silero VAD singleton ──────────────────────────────────────────────────────
#
# Used for post-hoc refinement of RMS-gated segments: trims leading/trailing
# silence that the energy-based VAD leaves in, recovering truncated word endings.

_silero_vad_model: Any | None = None
_silero_vad_lock = threading.Lock()


def init_silero_vad() -> Any | None:
    """Load Silero VAD model (idempotent). Returns model or None if unavailable."""
    global _silero_vad_model
    with _silero_vad_lock:
        if _silero_vad_model is not None:
            return _silero_vad_model
        try:
            import torch  # noqa: F401
            from silero_vad import load_silero_vad  # type: ignore[import-untyped]
            _silero_vad_model = load_silero_vad()
            return _silero_vad_model
        except Exception as exc:
            print(
                f"[{now_ts()}] [warn] Silero VAD unavailable, falling back to RMS-only trim: {exc}",
                flush=True,
            )
            return None


def _s16le_to_float(raw: bytes) -> list[float]:
    """Convert raw s16le PCM bytes to normalised float32 list expected by moonshine."""
    samples = array.array("h", raw)
    return [s / 32768.0 for s in samples]


def init_transcriber(model_size: str = "base") -> None:
    """Load the moonshine model into the module-level singleton (idempotent)."""
    global _transcriber
    with _transcriber_lock:
        if _transcriber is not None:
            return
        try:
            from moonshine_voice import Transcriber, get_model_for_language  # type: ignore[import-untyped]
            from moonshine_voice.moonshine_api import ModelArch  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "moonshine_voice not installed. Run: pip install moonshine-voice"
            ) from exc

        arch_map = {
            "tiny": ModelArch.TINY,
            "base": ModelArch.BASE,
        }
        if model_size not in arch_map:
            raise ValueError(f"--model-size must be 'tiny' or 'base', got {model_size!r}")

        arch = arch_map[model_size]
        print(f"[{now_ts()}] loading moonshine-{model_size}-ja ...", flush=True)
        model_path, model_arch = get_model_for_language("ja", arch)
        _transcriber = Transcriber(
            model_path=model_path,
            model_arch=model_arch,
            # max_tokens_per_second: moonshine default is tuned for English (~6).
            # Japanese token density is higher; 13.0 prevents early truncation.
            options={"max_tokens_per_second": 13.0},
        )
        print(f"[{now_ts()}] moonshine-{model_size}-ja ready", flush=True)


def _warm_up_transcriber() -> None:
    """Run a silent dummy inference to JIT-compile the model graph."""
    global _transcriber
    if _transcriber is None:
        return
    silence = [0.0] * SAMPLE_RATE  # 1 second of silence
    try:
        _transcriber.transcribe_without_streaming(silence, sample_rate=SAMPLE_RATE, flags=0)
        print(f"[{now_ts()}] moonshine warm-up done", flush=True)
    except Exception as exc:
        print(f"[{now_ts()}] moonshine warm-up warning: {exc}", flush=True)


def transcribe_with_server(
    wav_path: Path,
    *,
    base_url: str | None = None,           # ignored — kept for API compatibility
    language: str | None = None,           # ignored — model is ja-specific
    prompt: str | None = None,             # ignored — moonshine has no prompt injection
    response_format: str = "json",         # ignored
    temperature: str = "0.0",             # ignored
    temperature_inc: str = "0.2",         # ignored
    timeout_sec: int = 60,                # ignored
) -> str:
    """Transcribe a WAV file using the module-level moonshine Transcriber.

    The signature is intentionally compatible with listen_only_whisper_server so
    that voice_command_loop._transcribe_segment() works without modification.
    """
    global _transcriber, _stt_prompt_warned

    if _transcriber is None:
        raise RuntimeError(
            "moonshine transcriber not initialised; call init_transcriber() first"
        )

    if prompt and prompt.strip() and not _stt_prompt_warned:
        print(
            f"[{now_ts()}] NOTE: --stt-prompt is not supported by moonshine and will be ignored",
            flush=True,
        )
        _stt_prompt_warned = True

    try:
        from moonshine_voice import load_wav_file
        audio_data, sample_rate = load_wav_file(str(wav_path))
    except Exception as exc:
        raise RuntimeError(f"failed to load wav {wav_path}: {exc}") from exc

    with _transcriber_lock:
        result = _transcriber.transcribe_without_streaming(
            audio_data, sample_rate=sample_rate, flags=0
        )

    return " ".join(line.text for line in result.lines)


# ── ListenLoop ────────────────────────────────────────────────────────────────

class ListenLoop:
    """Microphone VAD + moonshine transcription loop.

    Identical public interface to listen_only_whisper_server.ListenLoop.
    The only behavioural differences are:
    - Audio is captured via ffmpeg (not parec).
    - Transcription is done in-process via moonshine (not via HTTP to whisper-server).
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop_requested = False

        if getattr(args, "run_command", None):
            self.source = getattr(args, "source", None) or "<run-command>"
        else:
            self.source = detect_source(getattr(args, "source", None))

        # base_url kept for API compat — not used for moonshine
        self.base_url = getattr(args, "server", None) or "moonshine://local"

        self.chunk_ms = args.chunk_ms
        self.chunk_samples = SAMPLE_RATE * self.chunk_ms // 1000
        self.chunk_bytes = self.chunk_samples * BYTES_PER_SAMPLE

        self.pre_roll_chunks = max(0, args.pre_roll_ms // self.chunk_ms)
        self.start_chunks = max(1, args.start_ms // self.chunk_ms)
        self.end_silence_chunks = max(1, args.end_silence_ms // self.chunk_ms)
        self.max_speech_chunks = max(1, args.max_speech_ms // self.chunk_ms)
        self.min_speech_chunks = max(1, args.min_speech_ms // self.chunk_ms)
        # [E] post-speech padding: after silence triggers segment end, read N extra
        # chunks from the audio stream before finalising the payload.  Trailing weak
        # phonemes (e.g. 'て' in 'にして') can arrive up to ~200ms after energy-VAD
        # silence detection; capturing them prevents moonshine from mis-transcribing
        # clipped word endings.  Silero VAD post-hoc trim will remove any genuine
        # silence that follows.
        self.post_speech_pad_chunks = max(0, getattr(args, "post_speech_pad_ms", 200) // self.chunk_ms)

        self.noise_rms_values: list[float] = []
        self.calibration_chunks = max(0, args.calibration_ms // self.chunk_ms)
        self.start_threshold = args.start_rms
        self.stop_threshold = args.stop_rms

        self.segments_seen = 0
        self.started_at = time.time()

        # Load models at init time (once per process)
        if not getattr(args, "run_command", None):
            model_size = getattr(args, "model_size", "base")
            init_transcriber(model_size)
            _warm_up_transcriber()
            # Silero VAD: post-hoc segment refinement (trims silence moonshine can't handle)
            self._silero_vad_model = init_silero_vad()
            if self._silero_vad_model is not None:
                self.log("Silero VAD loaded: post-hoc silence trimming enabled")
        else:
            self._silero_vad_model = None

    def log(self, msg: str) -> None:
        print(f"[{now_ts()}] {msg}", flush=True)

    def debug(self, msg: str) -> None:
        if self.args.debug:
            self.log(msg)

    def _update_thresholds_from_noise(self) -> None:
        if not self.noise_rms_values:
            return
        vals = sorted(self.noise_rms_values)
        median = vals[len(vals) // 2]
        self.start_threshold = max(
            self.args.start_rms_min,
            min(self.args.start_rms_max, median * 4.0),
        )
        self.stop_threshold = max(
            self.args.stop_rms_min,
            min(self.args.stop_rms_max, median * 2.2),
        )
        self.log(
            f"noise calibration median_rms={median:.4f} "
            f"-> start_rms={self.start_threshold:.4f}, stop_rms={self.stop_threshold:.4f}"
        )

    def _spawn_ffmpeg(self) -> subprocess.Popen[bytes]:
        """Spawn ffmpeg reading from PulseAudio source, writing raw s16le PCM to stdout.

        ffmpeg is used instead of parec because:
        1. It correctly captures PipeWire virtual sources (including DJI MIC MINI).
        2. It handles sample-rate conversion internally (always outputs 16 kHz).
        3. It is already a hard dependency of the surrounding system.
        """
        env = os.environ.copy()
        if "XDG_RUNTIME_DIR" not in env:
            candidate = f"/run/user/{os.getuid()}"
            if os.path.isdir(candidate):
                env["XDG_RUNTIME_DIR"] = candidate

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "pulse",
            "-i", self.source,
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-f", "s16le",
            "pipe:1",
        ]
        self.debug("spawning ffmpeg: " + " ".join(cmd))
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

    def _refine_with_silero(self, raw_pcm: bytes) -> bytes:
        """Post-process a raw PCM segment with Silero VAD.

        Finds the actual speech span and trims silence from both ends.
        This recovers weak trailing phonemes (e.g. 'て' in 'にして') that fall below
        the RMS stop_threshold and would otherwise be wrongly trimmed by the energy VAD.
        Falls back to raw_pcm on any error or when Silero VAD is unavailable.
        """
        if self._silero_vad_model is None:
            return raw_pcm
        try:
            import torch
            from silero_vad import get_speech_timestamps
            n = len(raw_pcm) // 2
            samples = struct.unpack(f"<{n}h", raw_pcm[: n * 2])
            audio = torch.tensor([s / 32768.0 for s in samples], dtype=torch.float32)
            with _silero_vad_lock:
                ts_list = get_speech_timestamps(
                    audio,
                    self._silero_vad_model,
                    sampling_rate=SAMPLE_RATE,
                    threshold=0.4,
                    min_silence_duration_ms=300,
                    speech_pad_ms=200,   # keep 200ms padding around speech edges
                )
            if not ts_list:
                self.debug("silero: no speech detected, using full segment")
                return raw_pcm
            start_byte = ts_list[0]["start"] * BYTES_PER_SAMPLE
            end_byte = min(ts_list[-1]["end"] * BYTES_PER_SAMPLE, len(raw_pcm))
            refined = raw_pcm[start_byte:end_byte]
            if len(refined) < self.min_speech_chunks * self.chunk_bytes:
                self.debug(f"silero: refined too short ({len(refined)}B), using full segment")
                return raw_pcm
            orig_sec = len(raw_pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE
            refined_sec = len(refined) / BYTES_PER_SAMPLE / SAMPLE_RATE
            pre_trim_sec = start_byte / BYTES_PER_SAMPLE / SAMPLE_RATE
            post_trim_sec = (len(raw_pcm) - end_byte) / BYTES_PER_SAMPLE / SAMPLE_RATE
            self.debug(
                f"silero: {orig_sec:.2f}s -> {refined_sec:.2f}s "
                f"(pre_trim={pre_trim_sec:.2f}s post_trim={post_trim_sec:.2f}s)"
            )
            return refined
        except Exception as exc:
            self.log(f"silero refinement error: {exc}")
            return raw_pcm

    def _handle_segment(self, raw_pcm: bytes, *, reason: str) -> None:
        if len(raw_pcm) < self.min_speech_chunks * self.chunk_bytes:
            self.debug(f"segment skipped (too short): {len(raw_pcm)} bytes")
            return

        # [D] Silero VAD post-hoc refinement: trim silence from both ends more accurately
        raw_pcm = self._refine_with_silero(raw_pcm)
        # Re-check length after possible trimming
        if len(raw_pcm) < self.min_speech_chunks * self.chunk_bytes:
            self.debug(f"segment skipped after silero trim (too short): {len(raw_pcm)} bytes")
            return

        self.segments_seen += 1
        seg_id = self.segments_seen
        tmp_dir = Path(self.args.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        wav_path = tmp_dir / f"moonshine-seg-{ts}-{seg_id:04d}.wav"
        wav_path.write_bytes(wav_bytes_from_pcm(raw_pcm))
        dur_sec = len(raw_pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE

        self.log(
            f"speech segment #{seg_id} captured ({dur_sec:.2f}s, reason={reason}) -> transcribing ..."
        )
        try:
            t0 = time.time()
            text = transcribe_with_server(
                wav_path,
                prompt=getattr(self.args, "stt_prompt", None),
            )
            elapsed = time.time() - t0
            text = " ".join(text.split())
            if text:
                self.log(f"transcript #{seg_id} ({elapsed:.2f}s): {text}")
            else:
                self.log(f"transcript #{seg_id} empty ({elapsed:.2f}s)")
        except Exception as exc:
            self.log(f"transcription error: {exc}")

    def run(self) -> int:
        self.log(f"source={self.source}")
        self.log(f"backend=moonshine model_size={getattr(self.args, 'model_size', 'base')}")

        proc = self._spawn_ffmpeg()
        assert proc.stdout is not None

        pre_roll: collections.deque[bytes] = collections.deque(maxlen=self.pre_roll_chunks)
        in_speech = False
        speech_buf = bytearray()
        speech_chunks = 0
        hot_count = 0
        quiet_count = 0
        chunks_seen = 0

        try:
            while not self.stop_requested:
                if self.args.max_run_sec > 0 and (time.time() - self.started_at) >= self.args.max_run_sec:
                    self.log(f"max_run_sec reached ({self.args.max_run_sec}s), stopping")
                    break

                chunk = proc.stdout.read(self.chunk_bytes)
                if not chunk or len(chunk) < self.chunk_bytes:
                    stderr_tail = b""
                    try:
                        assert proc.stderr is not None
                        stderr_tail = proc.stderr.read() or b""
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"ffmpeg ended unexpectedly "
                        f"(read={len(chunk) if chunk else 0}) "
                        + stderr_tail.decode(errors="ignore")
                    )

                chunks_seen += 1
                rms = pcm_rms(chunk)

                # ── Calibration phase ─────────────────────────────────────
                if self.calibration_chunks and chunks_seen <= self.calibration_chunks:
                    self.noise_rms_values.append(rms)
                    if chunks_seen == self.calibration_chunks:
                        self._update_thresholds_from_noise()
                    self.debug(f"calibration chunk={chunks_seen} rms={rms:.4f}")
                    pre_roll.append(chunk)
                    continue

                # ── Idle (not in speech) ──────────────────────────────────
                if not in_speech:
                    pre_roll.append(chunk)
                    if rms >= self.start_threshold:
                        hot_count += 1
                    else:
                        hot_count = 0

                    if hot_count >= self.start_chunks:
                        in_speech = True
                        quiet_count = 0
                        speech_chunks = 0
                        speech_buf = bytearray().join(pre_roll)
                        speech_chunks += len(pre_roll)
                        pre_roll.clear()
                        self.log(
                            f"speech start (rms={rms:.4f}, threshold={self.start_threshold:.4f})"
                        )
                    else:
                        self.debug(f"idle rms={rms:.4f}")
                    continue

                # ── In speech ─────────────────────────────────────────────
                speech_buf.extend(chunk)
                speech_chunks += 1

                if rms < self.stop_threshold:
                    quiet_count += 1
                else:
                    quiet_count = 0

                self.debug(
                    f"speech rms={rms:.4f} "
                    f"quiet_count={quiet_count}/{self.end_silence_chunks} "
                    f"speech_chunks={speech_chunks}"
                )

                if speech_chunks >= self.max_speech_chunks:
                    self._handle_segment(bytes(speech_buf), reason="max_speech")
                    in_speech = False
                    speech_buf = bytearray()
                    speech_chunks = 0
                    hot_count = 0
                    quiet_count = 0
                    if 0 < self.args.max_segments <= self.segments_seen:
                        self.log(f"max_segments reached ({self.args.max_segments}), stopping")
                        break
                    continue

                if quiet_count >= self.end_silence_chunks:
                    # [E] post-speech padding: read a few extra chunks from the stream
                    # BEFORE computing the tail-keep trim.  This captures trailing weak
                    # phonemes that arrive after energy-VAD silence detection fires.
                    for _pad_i in range(self.post_speech_pad_chunks):
                        _extra = proc.stdout.read(self.chunk_bytes)
                        if not _extra or len(_extra) < self.chunk_bytes:
                            break
                        speech_buf.extend(_extra)
                        quiet_count += 1

                    # [A] tail-keep padding: retain the last TAIL_KEEP_CHUNKS chunks
                    # instead of trimming all quiet chunks.  Weak trailing phonemes
                    # (e.g. 'て' in 'にして') often dip below stop_threshold; keeping
                    # 200ms of tail avoids cutting them off before Silero VAD refines.
                    # Silero VAD (_refine_with_silero) will remove genuine silence later.
                    TAIL_KEEP_CHUNKS = 2  # 200ms at chunk_ms=100
                    trim_chunks = max(0, quiet_count - TAIL_KEEP_CHUNKS)
                    trim_bytes = trim_chunks * self.chunk_bytes
                    payload = (
                        bytes(speech_buf[:-trim_bytes])
                        if trim_bytes > 0 and trim_bytes < len(speech_buf)
                        else bytes(speech_buf)
                    )
                    self._handle_segment(payload, reason="silence")
                    in_speech = False
                    speech_buf = bytearray()
                    speech_chunks = 0
                    hot_count = 0
                    quiet_count = 0
                    if 0 < self.args.max_segments <= self.segments_seen:
                        self.log(f"max_segments reached ({self.args.max_segments}), stopping")
                        break

        except KeyboardInterrupt:
            self.log("keyboard interrupt")
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        return 0


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Listen-only microphone transcription using moonshine"
    )
    # ── moonshine-specific ────────────────────────────────────────────────
    p.add_argument(
        "--model-size",
        default="base",
        choices=["tiny", "base"],
        help="moonshine model size: tiny (~170ms/seg) or base (~270ms/seg, default)",
    )
    # ── API-compatibility args (kept so voice_command_loop.py can pass them) ─
    p.add_argument(
        "--server",
        default=None,
        help="[ignored] whisper-server URL — not used with moonshine",
    )
    p.add_argument(
        "--language",
        default="ja",
        help="[ignored] language code — moonshine model is language-specific",
    )
    p.add_argument(
        "--stt-prompt",
        default="",
        help="[ignored] vocabulary hint — moonshine has no prompt injection",
    )
    p.add_argument(
        "--server-ready-timeout-sec",
        type=float,
        default=10.0,
        help="[ignored] not applicable to moonshine",
    )
    # ── audio / VAD ──────────────────────────────────────────────────────
    p.add_argument("--source", default=None, help="Pulse/PipeWire source name (auto-detect if omitted)")
    p.add_argument("--tmp-dir", default="/tmp/moonshine-listen-segments", help="Directory to write temp WAV segments")
    p.add_argument("--chunk-ms", type=int, default=100)
    p.add_argument("--pre-roll-ms", type=int, default=300)
    p.add_argument("--start-ms", type=int, default=200)
    p.add_argument("--end-silence-ms", type=int, default=800)
    p.add_argument("--min-speech-ms", type=int, default=350)
    p.add_argument("--max-speech-ms", type=int, default=10000)
    p.add_argument("--calibration-ms", type=int, default=800)
    p.add_argument("--start-rms", type=float, default=0.020)
    p.add_argument("--stop-rms", type=float, default=0.010)
    p.add_argument("--start-rms-min", type=float, default=0.010)
    p.add_argument("--start-rms-max", type=float, default=0.060)
    # [B] lower stop_rms_min vs whisper backend (0.006): Japanese word endings (e.g. 'て',
    # 'ね', 'よ') are weak and dip to ~0.003; allowing stop_rms down to 0.002 keeps them
    # in the segment until Silero VAD can refine boundaries accurately.
    p.add_argument("--stop-rms-min", type=float, default=0.002)
    p.add_argument("--stop-rms-max", type=float, default=0.040)
    p.add_argument(
        "--post-speech-pad-ms",
        type=int,
        default=200,
        help="Extra ms of audio to read after silence trigger, before tail-keep trim. "
             "Preserves trailing weak phonemes (default: 200 for moonshine)",
    )
    p.add_argument("--max-run-sec", type=int, default=0, help="0 means run forever")
    p.add_argument("--max-segments", type=int, default=0, help="0 means unlimited")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    loop = ListenLoop(args)

    def _sig_handler(signum, frame):  # noqa: ANN001
        loop.stop_requested = True
        loop.log(f"signal {signum} received, stopping ...")

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    return loop.run()


if __name__ == "__main__":
    raise SystemExit(main())
