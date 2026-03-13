from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ahear import collect_guest_voice as mod


def test_label_prefers_known_microphone_names() -> None:
    assert mod.label_for_source("alsa_input.usb-DJI_MIC_MINI-01.analog-stereo") == "dji"
    assert mod.label_for_source("alsa_input.usb-Razer_Seiren_Mini-00.analog-stereo") == "razer"
    assert mod.label_for_source("alsa_input.usb-HD_Pro_Webcam_C920-02.analog-stereo") == "webcam"


def test_label_falls_back_to_last_segment() -> None:
    assert mod.label_for_source("foo.bar.bazqux") == "bazqux"


def test_resolve_sources_skips_monitor_entries_and_deduplicates() -> None:
    with patch(
        "ahear.collect_guest_voice.list_pulse_sources",
        return_value=[
            "alsa_input.usb-DJI_MIC_MINI-01.analog-stereo",
            "alsa_input.usb-DJI_MIC_MINI-01.analog-stereo",
            "alsa_output.pci-0000.monitor",
            "alsa_input.usb-Razer_Seiren_Mini-00.analog-stereo",
        ],
    ):
        result = mod.resolve_sources(["dji", "razer", "dji"])

    assert result == [
        ("alsa_input.usb-DJI_MIC_MINI-01.analog-stereo", "dji"),
        ("alsa_input.usb-Razer_Seiren_Mini-00.analog-stereo", "razer"),
    ]


def test_resolve_sources_without_keywords_returns_all_non_monitor_sources() -> None:
    with patch(
        "ahear.collect_guest_voice.list_pulse_sources",
        return_value=[
            "alsa_output.pci-0000.monitor",
            "alsa_input.usb-DJI_MIC_MINI-01.analog-stereo",
            "alsa_input.usb-Razer_Seiren_Mini-00.analog-stereo",
        ],
    ):
        result = mod.resolve_sources([])

    assert result == [
        ("alsa_input.usb-DJI_MIC_MINI-01.analog-stereo", "dji"),
        ("alsa_input.usb-Razer_Seiren_Mini-00.analog-stereo", "razer"),
    ]


def test_arg_parser_defaults_to_persistent_guest_voice_dir() -> None:
    args = mod.build_arg_parser().parse_args([])
    assert args.out_dir == str(mod.DEFAULT_OUT_DIR)
    assert args.sources == "dji,razer"
