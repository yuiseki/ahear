"""Guest voice collector with simple energy VAD."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2
BYTES_PER_SAMPLE = CHANNELS * SAMPLE_WIDTH

CHUNK_MS = 80
START_RMS = 0.012
STOP_RMS = 0.004
START_CHUNKS = 2
END_SILENCE_CHUNKS = 8
MIN_SPEECH_SEC = 0.4
MAX_SPEECH_SEC = 15.0
PRE_ROLL_CHUNKS = 3

DEFAULT_SOURCES = "dji,razer"
DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "AHEAR_GUEST_VOICE_DIR",
        "/home/yuiseki/Workspaces/private/datasets/voices/others",
    )
)


def pcm_rms(chunk: bytes) -> float:
    sample_count = len(chunk) // 2
    if sample_count <= 0:
        return 0.0
    samples = struct.unpack("<" + "h" * sample_count, chunk[: sample_count * 2])
    mean_sq = sum(sample * sample for sample in samples) / len(samples)
    return math.sqrt(mean_sq) / 32768.0


def make_wav(raw_pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
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
        1,
        CHANNELS,
        sample_rate,
        byte_rate,
        block_align,
        SAMPLE_WIDTH * 8,
        b"data",
        data_size,
    )
    return header + raw_pcm


def list_pulse_sources() -> list[str]:
    try:
        output = subprocess.check_output(
            ["pactl", "list", "short", "sources"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [line.split("\t")[1] for line in output.strip().splitlines() if "\t" in line]


def label_for_source(source_name: str) -> str:
    lowered = source_name.lower()
    if "dji" in lowered:
        return "dji"
    if "razer" in lowered:
        return "razer"
    if "c920" in lowered or "webcam" in lowered:
        return "webcam"
    parts = source_name.split(".")
    return parts[-1][:20] if parts else source_name[:20]


def resolve_sources(keywords: Sequence[str]) -> list[tuple[str, str]]:
    all_sources = [name for name in list_pulse_sources() if "monitor" not in name.lower()]
    if not keywords:
        return [(source, label_for_source(source)) for source in all_sources]

    resolved: list[tuple[str, str]] = []
    seen_sources: set[str] = set()
    for keyword in keywords:
        matches = [source for source in all_sources if keyword.lower() in source.lower()]
        if not matches:
            print(f"[warn] no source matching '{keyword}', skipping", file=sys.stderr)
        for source in matches:
            if source in seen_sources:
                continue
            resolved.append((source, label_for_source(source)))
            seen_sources.add(source)
    return resolved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect guest voice samples with energy VAD")
    parser.add_argument(
        "--sources",
        default=DEFAULT_SOURCES,
        help="Comma-separated keywords to match PulseAudio source names",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for guest voice WAVs",
    )
    return parser


class MicRecorder:
    def __init__(
        self,
        *,
        source: str,
        label: str,
        out_dir: Path,
        stop_event: threading.Event,
    ) -> None:
        self.source = source
        self.label = label
        self.out_dir = out_dir / label
        self.stop_event = stop_event
        self.seq = 0
        self.saved = 0
        chunk_samples = SAMPLE_RATE * CHUNK_MS // 1000
        self.chunk_bytes = chunk_samples * BYTES_PER_SAMPLE
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _spawn_ffmpeg(self) -> subprocess.Popen[bytes]:
        env = os.environ.copy()
        if "XDG_RUNTIME_DIR" not in env:
            candidate = f"/run/user/{os.getuid()}"
            if os.path.isdir(candidate):
                env["XDG_RUNTIME_DIR"] = candidate
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "pulse",
            "-i",
            self.source,
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            str(CHANNELS),
            "-f",
            "s16le",
            "pipe:1",
        ]
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._record_loop()
            except Exception as exc:
                print(f"[{self.label}] error: {exc}, restarting in 2s", file=sys.stderr)
                time.sleep(2)

    def _record_loop(self) -> None:
        process = self._spawn_ffmpeg()
        pre_roll: deque[bytes] = deque(maxlen=PRE_ROLL_CHUNKS)
        speech_buf: list[bytes] = []
        in_speech = False
        onset_count = 0
        quiet_count = 0
        min_bytes = int(MIN_SPEECH_SEC * SAMPLE_RATE * BYTES_PER_SAMPLE)
        max_bytes = int(MAX_SPEECH_SEC * SAMPLE_RATE * BYTES_PER_SAMPLE)

        try:
            while not self.stop_event.is_set():
                if process.stdout is None:
                    break
                chunk = process.stdout.read(self.chunk_bytes)
                if not chunk:
                    break
                rms = pcm_rms(chunk)

                if not in_speech:
                    pre_roll.append(chunk)
                    if rms >= START_RMS:
                        onset_count += 1
                        if onset_count >= START_CHUNKS:
                            in_speech = True
                            onset_count = 0
                            quiet_count = 0
                            speech_buf = list(pre_roll)
                    else:
                        onset_count = 0
                    continue

                speech_buf.append(chunk)
                if rms < STOP_RMS:
                    quiet_count += 1
                else:
                    quiet_count = 0

                total_bytes = sum(len(part) for part in speech_buf)
                if quiet_count < END_SILENCE_CHUNKS and total_bytes < max_bytes:
                    continue

                in_speech = False
                raw_pcm = b"".join(speech_buf)
                speech_buf = []
                pre_roll.clear()
                if len(raw_pcm) >= min_bytes:
                    self._save(raw_pcm)
        finally:
            process.terminate()
            process.wait()

    def _save(self, raw_pcm: bytes) -> None:
        self.seq += 1
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = self.out_dir / f"guest-{timestamp}-{self.seq:04d}.wav"
        filename.write_bytes(make_wav(raw_pcm))
        duration = len(raw_pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE
        self.saved += 1
        print(f"[{self.label}] saved #{self.seq}: {filename.name} ({duration:.2f}s)")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    keywords = [item.strip() for item in str(args.sources).split(",") if item.strip()]
    sources = resolve_sources(keywords)
    if not sources:
        print("ERROR: no matching sources found", file=sys.stderr)
        print("Available sources:", file=sys.stderr)
        for source in list_pulse_sources():
            print(f"  {source}", file=sys.stderr)
        return 1

    out_dir = Path(str(args.out_dir))
    stop_event = threading.Event()
    recorders = [
        MicRecorder(source=source, label=label, out_dir=out_dir, stop_event=stop_event)
        for source, label in sources
    ]
    threads = [
        threading.Thread(target=recorder.run, daemon=True, name=f"guest-voice-{recorder.label}")
        for recorder in recorders
    ]
    print(f"Recording from {len(recorders)} source(s). Press Ctrl+C to stop.")
    for thread in threads:
        thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
    for thread in threads:
        thread.join(timeout=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
