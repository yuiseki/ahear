from __future__ import annotations

import datetime
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ahear import whisper_listener as mod


def _patch_sources(names: list[str]):
    return patch("ahear.whisper_listener.list_sources", return_value=names)


class TestDetectSourceExclusions:
    def test_dji_mic_mini_is_selected(self) -> None:
        sources = [
            "alsa_input.usb-DJI_Technology_Co.__Ltd._DJI_MIC_MINI_XSP12345678B-01.analog-stereo",
            "alsa_input.usb-046d_HD_Pro_Webcam_C920_451C46DF-02.analog-stereo",
            "alsa_input.pci-0000_0c_00.4.pro-input-0",
        ]
        with _patch_sources(sources):
            result = mod.detect_source(None)
        assert "DJI_MIC_MINI" in result

    def test_c920_webcam_is_excluded(self) -> None:
        sources = [
            "alsa_input.usb-046d_HD_Pro_Webcam_C920_451C46DF-02.analog-stereo",
            "alsa_input.usb-046d_HD_Pro_Webcam_C920_8DB9885F-02.analog-stereo",
        ]
        with _patch_sources(sources):
            with pytest.raises(RuntimeError, match="No usable"):
                mod.detect_source(None)

    def test_pci_line_input_is_excluded(self) -> None:
        sources = [
            "alsa_input.pci-0000_0c_00.4.pro-input-0",
            "alsa_input.pci-0000_0c_00.4.pro-input-2",
        ]
        with _patch_sources(sources):
            with pytest.raises(RuntimeError, match="No usable"):
                mod.detect_source(None)

    def test_monitor_sources_are_excluded(self) -> None:
        sources = [
            "alsa_output.pci-0000_04_00.1.pro-output-3.monitor",
            "vacuumtube_silent.monitor",
        ]
        with _patch_sources(sources):
            with pytest.raises(RuntimeError, match="No usable"):
                mod.detect_source(None)

    def test_razer_selected_when_no_dji(self) -> None:
        sources = [
            "alsa_input.usb-Razer_Seiren_Mini_00000000-00.analog-stereo",
            "alsa_input.usb-046d_HD_Pro_Webcam_C920_451C46DF-02.analog-stereo",
            "alsa_input.pci-0000_0c_00.4.pro-input-0",
        ]
        with _patch_sources(sources):
            result = mod.detect_source(None)
        assert "Razer_Seiren_Mini" in result

    def test_unknown_usb_mic_selected_as_last_resort(self) -> None:
        sources = [
            "alsa_input.usb-AONI_HD_webcam-CMS-V43BK-02.mono-fallback",
            "alsa_input.pci-0000_0c_00.4.pro-input-0",
        ]
        with _patch_sources(sources):
            result = mod.detect_source(None)
        assert "AONI" in result

    def test_no_mic_raises(self) -> None:
        sources = [
            "alsa_output.pci-0000_04_00.1.pro-output-3.monitor",
            "alsa_input.usb-046d_HD_Pro_Webcam_C920_451C46DF-02.analog-stereo",
            "alsa_input.pci-0000_0c_00.4.pro-input-0",
            "vacuumtube_silent.monitor",
        ]
        with _patch_sources(sources):
            with pytest.raises(RuntimeError, match="No usable"):
                mod.detect_source(None)

    def test_user_source_bypasses_detection(self) -> None:
        with _patch_sources([]):
            result = mod.detect_source("custom_source_name")
        assert result == "custom_source_name"


def _make_server(tmp_path: Path) -> mod.ListenLoop:
    args = MagicMock()
    args.tmp_dir = str(tmp_path)
    args.server = "http://127.0.0.1:18080"
    args.language = "ja"
    args.prompt = ""
    args.debug = False
    server = mod.ListenLoop.__new__(mod.ListenLoop)
    server.args = args
    server.segments_seen = 0
    server.chunk_bytes = 2
    server.min_speech_chunks = 0
    server.log = lambda *a, **kw: None
    server.debug = lambda *a, **kw: None
    server.base_url = args.server
    return server


class TestSegmentSavePath:
    def _fake_pcm(self, seconds: float = 0.5) -> bytes:
        n_samples = int(16000 * seconds)
        return b"\x00\x01" * n_samples

    def test_segment_saved_under_yyyy_mm_dd_hh_mm(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        fixed_dt = datetime.datetime(2026, 3, 5, 7, 42, 15)

        with patch("ahear.whisper_listener.transcribe_with_server", return_value="テスト"):
            with patch("ahear.whisper_listener.dt") as mock_dt_mod:
                mock_dt_mod.datetime.now.return_value = fixed_dt
                server._handle_segment(self._fake_pcm(), reason="vad")

        wavs = list(tmp_path.rglob("*.wav"))
        assert len(wavs) == 1
        rel = wavs[0].relative_to(tmp_path)
        parts = rel.parts
        assert len(parts) == 6
        assert parts[:5] == ("2026", "03", "05", "07", "42")

    def test_segment_filename_unchanged(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        fixed_dt = datetime.datetime(2026, 3, 5, 7, 42, 15)

        with patch("ahear.whisper_listener.transcribe_with_server", return_value="テスト"):
            with patch("ahear.whisper_listener.dt") as mock_dt_mod:
                mock_dt_mod.datetime.now.return_value = fixed_dt
                server._handle_segment(self._fake_pcm(), reason="vad")

        wavs = list(tmp_path.rglob("*.wav"))
        assert wavs[0].name.startswith("listen-seg-")

    def test_different_hours_go_to_different_dirs(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)

        for hour in [6, 7]:
            fixed_dt = datetime.datetime(2026, 3, 5, hour, 0, 0)
            with patch("ahear.whisper_listener.transcribe_with_server", return_value="x"):
                with patch("ahear.whisper_listener.dt") as mock_dt_mod:
                    mock_dt_mod.datetime.now.return_value = fixed_dt
                    server._handle_segment(self._fake_pcm(), reason="vad")

        wavs = list(tmp_path.rglob("*.wav"))
        assert len(wavs) == 2
        hour_dirs = {w.parent.parent.name for w in wavs}
        assert hour_dirs == {"06", "07"}
