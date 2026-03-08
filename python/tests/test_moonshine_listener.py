from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ahear import moonshine_listener as mod


def _make_loop(*, post_speech_pad_ms: int = 200, chunk_ms: int = 100) -> mod.ListenLoop:
    args = SimpleNamespace(
        chunk_ms=chunk_ms,
        pre_roll_ms=300,
        start_ms=200,
        end_silence_ms=800,
        min_speech_ms=350,
        max_speech_ms=10000,
        post_speech_pad_ms=post_speech_pad_ms,
        calibration_ms=800,
        start_rms=0.020,
        stop_rms=0.010,
        start_rms_min=0.010,
        start_rms_max=0.060,
        stop_rms_min=0.002,
        stop_rms_max=0.040,
        max_run_sec=0,
        max_segments=0,
        source=None,
        tmp_dir="/tmp/moonshine-listen-test",
        model_size="base",
        debug=False,
    )
    loop = mod.ListenLoop.__new__(mod.ListenLoop)
    loop.args = args
    loop.chunk_ms = args.chunk_ms
    loop.chunk_samples = 16000 * args.chunk_ms // 1000
    loop.chunk_bytes = loop.chunk_samples * 2
    loop.pre_roll_chunks = max(0, args.pre_roll_ms // args.chunk_ms)
    loop.start_chunks = max(1, args.start_ms // args.chunk_ms)
    loop.end_silence_chunks = max(1, args.end_silence_ms // args.chunk_ms)
    loop.max_speech_chunks = max(1, args.max_speech_ms // args.chunk_ms)
    loop.min_speech_chunks = max(1, args.min_speech_ms // args.chunk_ms)
    loop.post_speech_pad_chunks = max(0, args.post_speech_pad_ms // args.chunk_ms)
    loop.segments_seen = 0
    loop.log = lambda *a, **kw: None
    loop.debug = lambda *a, **kw: None
    loop._refine_with_silero = lambda raw_pcm: raw_pcm
    return loop


class TestPostSpeechPad:
    def test_post_speech_pad_chunks_default_200ms(self) -> None:
        loop = _make_loop(post_speech_pad_ms=200, chunk_ms=100)
        assert loop.post_speech_pad_chunks == 2

    def test_post_speech_pad_chunks_zero(self) -> None:
        loop = _make_loop(post_speech_pad_ms=0, chunk_ms=100)
        assert loop.post_speech_pad_chunks == 0

    def test_post_speech_pad_chunks_300ms(self) -> None:
        loop = _make_loop(post_speech_pad_ms=300, chunk_ms=100)
        assert loop.post_speech_pad_chunks == 3

    def test_parse_args_default_post_speech_pad_ms(self) -> None:
        with patch("sys.argv", ["moonshine_listener.py"]):
            args = mod.parse_args()
        assert args.post_speech_pad_ms == 200


class TestSegmentSavePath:
    def _fake_pcm(self, seconds: float = 0.5) -> bytes:
        n_samples = int(16000 * seconds)
        return b"\x00\x01" * n_samples

    def test_segment_saved_under_tmp_dir_root(self, tmp_path: Path) -> None:
        loop = _make_loop()
        loop.args.tmp_dir = str(tmp_path)
        fixed_dt = datetime.datetime(2026, 3, 5, 7, 42, 15)

        with patch("ahear.moonshine_listener.transcribe_with_server", return_value="テスト"):
            with patch("ahear.moonshine_listener.dt") as mock_dt_mod:
                mock_dt_mod.datetime.now.return_value = fixed_dt
                loop._handle_segment(self._fake_pcm(), reason="vad")

        wavs = list(tmp_path.glob("*.wav"))
        assert len(wavs) == 1
        assert wavs[0].name == "moonshine-seg-20260305-074215-0001.wav"
