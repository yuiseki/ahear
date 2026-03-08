#!/usr/bin/env python3
"""Listen-only microphone transcription using parec + simple energy VAD + whisper-server.

This is an experiment script for the current Ubuntu/KDE desktop:
- capture from PipeWire/PulseAudio source (USB mic preferred)
- detect utterances with RMS threshold (simple VAD)
- send WAV chunks to whisper.cpp whisper-server /inference
- print transcript lines

No response generation / TTS yet.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path


SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # s16le
BYTES_PER_SAMPLE = CHANNELS * SAMPLE_WIDTH


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


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
    # Minimal RIFF/WAV writer for PCM16 mono
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
        16,  # fmt chunk size
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
    cp = run(["pactl", "list", "short", "sources"], check=True)
    names: list[str] = []
    for line in cp.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            names.append(parts[1])
    return names


_EXCLUDED_PATTERNS = (
    ".monitor",   # speaker loopback
    "C920",       # Logitech webcam mic
    "alsa_input.pci-",  # motherboard line inputs
)


def _is_usable_mic(name: str) -> bool:
    """Return True if the source name represents a real, dedicated microphone."""
    return not any(pat in name for pat in _EXCLUDED_PATTERNS)


def detect_source(user_source: str | None) -> str:
    if user_source:
        return user_source
    names = list_sources()
    # Prefer DJI wireless mic if present (current owner's preferred mic).
    for n in names:
        if "DJI_MIC_MINI" in n:
            return n
    # Prefer Razer mic if present
    for n in names:
        if "Razer_Seiren_Mini" in n:
            return n
    # Any other dedicated USB mic (exclude webcams and line inputs)
    for n in names:
        if n.startswith("alsa_input.usb-") and _is_usable_mic(n):
            return n
    raise RuntimeError("No usable microphone found (webcam mics and line inputs are excluded)")


def wait_server_ready(base_url: str, timeout_sec: float = 10.0) -> None:
    # /inference requires multipart; use root page as a cheap readiness check.
    deadline = time.time() + timeout_sec
    curl_cmd = [
        "curl",
        "-fsS",
        "--max-time",
        "2",
        base_url.rstrip("/") + "/",
    ]
    last_err = ""
    while time.time() < deadline:
        cp = subprocess.run(curl_cmd, text=True, capture_output=True)
        if cp.returncode == 0:
            return
        last_err = (cp.stderr or cp.stdout or "").strip()
        time.sleep(0.3)
    raise RuntimeError(f"whisper-server not ready at {base_url}: {last_err}")


def transcribe_with_server(
    wav_path: Path,
    *,
    base_url: str,
    language: str,
    prompt: str | None = None,
    response_format: str = "json",
    temperature: str = "0.0",
    temperature_inc: str = "0.2",
    timeout_sec: int = 60,
) -> str:
    url = base_url.rstrip("/") + "/inference"
    cmd = [
        "curl",
        "-fsS",
        "--max-time",
        str(timeout_sec),
        url,
        "-H",
        "Content-Type: multipart/form-data",
        "-F",
        f"file=@{wav_path}",
        "-F",
        f"language={language}",
    ]
    if prompt and prompt.strip():
        cmd += [
            "-F",
            f"prompt={prompt.strip()}",
        ]
    cmd += [
        "-F",
        f"response_format={response_format}",
        "-F",
        f"temperature={temperature}",
        "-F",
        f"temperature_inc={temperature_inc}",
    ]
    cp = run(cmd, check=True)
    text = cp.stdout.strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return str(data.get("text", "")).strip()
    except json.JSONDecodeError:
        pass
    return text.strip()


class ListenLoop:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop_requested = False
        if getattr(args, "run_command", None):
            self.source = args.source or "<run-command>"
        else:
            self.source = detect_source(args.source)
        self.base_url = args.server.rstrip("/")

        self.chunk_ms = args.chunk_ms
        self.chunk_samples = SAMPLE_RATE * self.chunk_ms // 1000
        self.chunk_bytes = self.chunk_samples * BYTES_PER_SAMPLE

        self.pre_roll_chunks = max(0, args.pre_roll_ms // self.chunk_ms)
        self.start_chunks = max(1, args.start_ms // self.chunk_ms)
        self.end_silence_chunks = max(1, args.end_silence_ms // self.chunk_ms)
        self.max_speech_chunks = max(1, args.max_speech_ms // self.chunk_ms)
        self.min_speech_chunks = max(1, args.min_speech_ms // self.chunk_ms)

        self.noise_rms_values: list[float] = []
        self.calibration_chunks = max(0, args.calibration_ms // self.chunk_ms)
        self.start_threshold = args.start_rms
        self.stop_threshold = args.stop_rms

        self.segments_seen = 0
        self.started_at = time.time()

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
        # Conservative scaling, but keep sane floor/ceiling.
        self.start_threshold = max(self.args.start_rms_min, min(self.args.start_rms_max, median * 4.0))
        self.stop_threshold = max(self.args.stop_rms_min, min(self.args.stop_rms_max, median * 2.2))
        self.log(
            f"noise calibration median_rms={median:.4f} -> start_rms={self.start_threshold:.4f}, stop_rms={self.stop_threshold:.4f}"
        )

    def _spawn_parec(self) -> subprocess.Popen[bytes]:
        env = os.environ.copy()
        if "XDG_RUNTIME_DIR" not in env:
            candidate = f"/run/user/{os.getuid()}"
            if os.path.isdir(candidate):
                env["XDG_RUNTIME_DIR"] = candidate
        cmd = [
            "parec",
            "-d",
            self.source,
            "--format=s16le",
            "--channels=1",
            "--rate=16000",
            "--latency-msec=60",
        ]
        self.debug("spawning parec: " + " ".join(cmd))
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

    def _handle_segment(self, raw_pcm: bytes, *, reason: str) -> None:
        if len(raw_pcm) < self.min_speech_chunks * self.chunk_bytes:
            self.debug(f"segment skipped (too short): {len(raw_pcm)} bytes")
            return

        self.segments_seen += 1
        seg_id = self.segments_seen
        now = dt.datetime.now()
        tmp_dir = (
            Path(self.args.tmp_dir)
            / now.strftime("%Y")
            / now.strftime("%m")
            / now.strftime("%d")
            / now.strftime("%H")
            / now.strftime("%M")
        )
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ts = now.strftime("%Y%m%d-%H%M%S")
        wav_path = tmp_dir / f"listen-seg-{ts}-{seg_id:04d}.wav"
        wav_path.write_bytes(wav_bytes_from_pcm(raw_pcm))
        dur_sec = len(raw_pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE

        self.log(f"speech segment #{seg_id} captured ({dur_sec:.2f}s, reason={reason}) -> transcribing ...")
        try:
            t0 = time.time()
            text = transcribe_with_server(
                wav_path,
                base_url=self.base_url,
                language=self.args.language,
                prompt=getattr(self.args, "stt_prompt", None),
                response_format="json",
            )
            elapsed = time.time() - t0
            text = " ".join(text.split())
            if text:
                self.log(f"transcript #{seg_id} ({elapsed:.2f}s): {text}")
            else:
                self.log(f"transcript #{seg_id} empty ({elapsed:.2f}s)")
        except subprocess.CalledProcessError as e:
            self.log(f"transcription error (curl exit={e.returncode}): {(e.stderr or e.stdout or '').strip()}")
        except Exception as e:  # pragma: no cover - experiment script
            self.log(f"transcription error: {e}")

    def run(self) -> int:
        self.log(f"source={self.source}")
        self.log(f"server={self.base_url} language={self.args.language}")
        if getattr(self.args, "stt_prompt", ""):
            prompt_len = len(str(self.args.stt_prompt))
            self.log(f"stt prompt enabled ({prompt_len} chars)")
        self.log("checking whisper-server readiness ...")
        wait_server_ready(self.base_url, timeout_sec=self.args.server_ready_timeout_sec)
        self.log("whisper-server ready")

        proc = self._spawn_parec()
        assert proc.stdout is not None

        pre_roll = collections.deque(maxlen=self.pre_roll_chunks)
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
                    raise RuntimeError(f"parec ended unexpectedly (read={len(chunk) if chunk else 0}) {stderr_tail.decode(errors='ignore')}")

                chunks_seen += 1
                rms = pcm_rms(chunk)

                if self.calibration_chunks and chunks_seen <= self.calibration_chunks:
                    self.noise_rms_values.append(rms)
                    if chunks_seen == self.calibration_chunks:
                        self._update_thresholds_from_noise()
                    self.debug(f"calibration chunk={chunks_seen} rms={rms:.4f}")
                    pre_roll.append(chunk)
                    continue

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
                        speech_buf = bytearray().join(pre_roll)  # include pre-roll
                        speech_chunks += len(pre_roll)
                        pre_roll.clear()
                        self.log(f"speech start (rms={rms:.4f}, threshold={self.start_threshold:.4f})")
                    else:
                        self.debug(f"idle rms={rms:.4f}")
                    continue

                # in speech
                speech_buf.extend(chunk)
                speech_chunks += 1

                if rms < self.stop_threshold:
                    quiet_count += 1
                else:
                    quiet_count = 0

                self.debug(
                    f"speech rms={rms:.4f} quiet_count={quiet_count}/{self.end_silence_chunks} speech_chunks={speech_chunks}"
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
                    # Trim trailing silence chunks for cleaner clips.
                    trim_bytes = quiet_count * self.chunk_bytes
                    payload = bytes(speech_buf[:-trim_bytes]) if trim_bytes < len(speech_buf) else bytes(speech_buf)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Listen-only microphone transcription using whisper-server")
    p.add_argument("--server", default="http://127.0.0.1:18080", help="whisper-server base URL")
    p.add_argument("--language", default="ja", help="Whisper language code")
    p.add_argument("--source", default=None, help="Pulse/PipeWire source name (auto-detect if omitted)")
    p.add_argument("--tmp-dir", default="/tmp/whisper-listen-segments", help="Directory to write temp WAV segments")
    p.add_argument("--stt-prompt", default="", help="Optional initial prompt (vocabulary hint) passed to whisper-server /inference")
    p.add_argument("--chunk-ms", type=int, default=100)
    p.add_argument("--pre-roll-ms", type=int, default=300)
    p.add_argument("--start-ms", type=int, default=200)
    p.add_argument("--end-silence-ms", type=int, default=800)
    p.add_argument("--min-speech-ms", type=int, default=350)
    p.add_argument("--max-speech-ms", type=int, default=10000)
    p.add_argument("--calibration-ms", type=int, default=800)
    p.add_argument("--start-rms", type=float, default=0.020, help="initial start threshold before calibration")
    p.add_argument("--stop-rms", type=float, default=0.010, help="initial stop threshold before calibration")
    p.add_argument("--start-rms-min", type=float, default=0.010)
    p.add_argument("--start-rms-max", type=float, default=0.060)
    p.add_argument("--stop-rms-min", type=float, default=0.006)
    p.add_argument("--stop-rms-max", type=float, default=0.040)
    p.add_argument("--server-ready-timeout-sec", type=float, default=10.0)
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
