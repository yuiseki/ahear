from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ahear.speaker_auth import (
    SpeakerAuthRuntime,
    initialize_speaker_auth_runtime,
    speaker_auth_enabled,
    verify_speaker_identity,
)


def _build_auth_context(*, similarity: float, threshold: float = 0.5):
    fake_emb_np = np.array([1.0, 0.0, 0.0])
    bounded_similarity = float(max(-1.0, min(1.0, similarity)))
    voiceprint = np.array(
        [bounded_similarity, (max(0.0, 1.0 - bounded_similarity * bounded_similarity)) ** 0.5, 0.0]
    )
    voiceprint = voiceprint / np.linalg.norm(voiceprint)

    torchaudio_module = mock.MagicMock()
    fake_signal = mock.MagicMock()
    torchaudio_module.load.return_value = (fake_signal, 16000)

    torch_module = mock.MagicMock()
    no_grad = mock.MagicMock()
    no_grad.__enter__ = mock.Mock(return_value=None)
    no_grad.__exit__ = mock.Mock(return_value=None)
    torch_module.no_grad.return_value = no_grad

    classifier = mock.MagicMock()
    fake_emb_tensor = mock.MagicMock()
    fake_emb_tensor.squeeze.return_value.cpu.return_value.numpy.return_value = fake_emb_np
    classifier.encode_batch.return_value = fake_emb_tensor

    return classifier, voiceprint, torchaudio_module, torch_module, np, threshold


def test_speaker_auth_enabled_requires_classifier_and_voiceprint() -> None:
    assert speaker_auth_enabled(classifier=object(), voiceprint=object()) is True
    assert speaker_auth_enabled(classifier=None, voiceprint=object()) is False
    assert speaker_auth_enabled(classifier=object(), voiceprint=None) is False


def test_verify_speaker_identity_passes_above_threshold() -> None:
    classifier, voiceprint, torchaudio_module, torch_module, np_module, threshold = _build_auth_context(
        similarity=0.9,
        threshold=0.5,
    )
    logs: list[str] = []

    ok, err = verify_speaker_identity(
        wav_path=Path("/tmp/speaker.wav"),
        classifier=classifier,
        voiceprint=voiceprint,
        torchaudio_module=torchaudio_module,
        torch_module=torch_module,
        np_module=np_module,
        device="cpu",
        threshold=threshold,
        topk=5,
        auth_error_text="声紋認証に失敗しました",
        logger=logs.append,
        log_label="test",
        intent="system_status_report",
    )

    assert ok is True
    assert err is None
    assert any("AUTH PASSED" in line for line in logs)


def test_verify_speaker_identity_fails_below_threshold() -> None:
    classifier, voiceprint, torchaudio_module, torch_module, np_module, threshold = _build_auth_context(
        similarity=0.1,
        threshold=0.5,
    )

    ok, err = verify_speaker_identity(
        wav_path=Path("/tmp/speaker.wav"),
        classifier=classifier,
        voiceprint=voiceprint,
        torchaudio_module=torchaudio_module,
        torch_module=torch_module,
        np_module=np_module,
        device="cpu",
        threshold=threshold,
        topk=5,
        auth_error_text="声紋認証に失敗しました",
        logger=lambda _msg: None,
        log_label="test",
        intent="system_status_report",
    )

    assert ok is False
    assert err == "声紋認証に失敗しました"


def test_verify_speaker_identity_uses_topk_mean_for_2d_voiceprint() -> None:
    classifier, _voiceprint, torchaudio_module, torch_module, np_module, threshold = _build_auth_context(
        similarity=0.9,
        threshold=0.4,
    )
    voiceprint = np.array(
        [
            [0.9, (1.0 - 0.9 * 0.9) ** 0.5, 0.0],
            [0.8, (1.0 - 0.8 * 0.8) ** 0.5, 0.0],
            [0.1, (1.0 - 0.1 * 0.1) ** 0.5, 0.0],
        ]
    )
    voiceprint = voiceprint / np.linalg.norm(voiceprint, axis=1, keepdims=True)

    ok, err = verify_speaker_identity(
        wav_path=Path("/tmp/speaker.wav"),
        classifier=classifier,
        voiceprint=voiceprint,
        torchaudio_module=torchaudio_module,
        torch_module=torch_module,
        np_module=np_module,
        device="cpu",
        threshold=threshold,
        topk=2,
        auth_error_text="声紋認証に失敗しました",
        logger=lambda _msg: None,
        log_label="test",
        intent="system_status_report",
    )

    assert ok is True
    assert err is None


def test_verify_speaker_identity_returns_error_on_exception() -> None:
    classifier, voiceprint, torchaudio_module, torch_module, np_module, threshold = _build_auth_context(
        similarity=0.9,
        threshold=0.5,
    )
    torchaudio_module.load.side_effect = RuntimeError("torchaudio error")
    logs: list[str] = []

    ok, err = verify_speaker_identity(
        wav_path=Path("/tmp/speaker.wav"),
        classifier=classifier,
        voiceprint=voiceprint,
        torchaudio_module=torchaudio_module,
        torch_module=torch_module,
        np_module=np_module,
        device="cpu",
        threshold=threshold,
        topk=5,
        auth_error_text="声紋認証に失敗しました",
        logger=logs.append,
        log_label="test",
        intent="system_status_report",
    )

    assert ok is False
    assert err == "声紋認証に失敗しました"
    assert logs == ["Speaker ID verification error: torchaudio error"]


def test_initialize_speaker_auth_runtime_returns_empty_state_when_disabled() -> None:
    runtime = initialize_speaker_auth_runtime(
        enabled=False,
        requested_device="cpu",
        speaker_master="/tmp/master.npy",
        logger=lambda _msg: None,
    )

    assert runtime == SpeakerAuthRuntime(device="cpu")


def test_initialize_speaker_auth_runtime_loads_classifier_and_voiceprint() -> None:
    logs: list[str] = []
    classifier = object()
    voiceprint = np.array([1.0, 0.0, 0.0])
    torch_module = mock.Mock()
    torch_module.cuda.is_available.return_value = True
    np_module = mock.Mock()
    np_module.load.return_value = voiceprint
    torchaudio_module = mock.Mock()

    runtime = initialize_speaker_auth_runtime(
        enabled=True,
        requested_device="cpu",
        speaker_master="/tmp/master.npy",
        logger=logs.append,
        torchaudio_module=torchaudio_module,
        torch_module=torch_module,
        np_module=np_module,
        classifier_factory=mock.Mock(return_value=classifier),
        path_exists=lambda path: path == "/tmp/master.npy",
        load_voiceprint=np_module.load,
    )

    assert runtime.classifier is classifier
    assert runtime.voiceprint is voiceprint
    assert runtime.torchaudio_module is torchaudio_module
    assert runtime.torch_module is torch_module
    assert runtime.np_module is np_module
    assert runtime.device == "cuda:0"
    assert any("CUDA detected, upgrading device to cuda:0" in line for line in logs)
    assert any("master voiceprint loaded from /tmp/master.npy" in line for line in logs)


def test_initialize_speaker_auth_runtime_logs_missing_master_voiceprint() -> None:
    logs: list[str] = []

    runtime = initialize_speaker_auth_runtime(
        enabled=True,
        requested_device="cpu",
        speaker_master="/tmp/missing.npy",
        logger=logs.append,
        torchaudio_module=mock.Mock(),
        torch_module=mock.Mock(cuda=mock.Mock(is_available=mock.Mock(return_value=False))),
        np_module=mock.Mock(),
        classifier_factory=mock.Mock(return_value=object()),
        path_exists=lambda _path: False,
        load_voiceprint=mock.Mock(),
    )

    assert runtime.voiceprint is None
    assert any("master voiceprint not found" in line for line in logs)


def test_initialize_speaker_auth_runtime_logs_import_error() -> None:
    logs: list[str] = []

    runtime = initialize_speaker_auth_runtime(
        enabled=True,
        requested_device="cpu",
        speaker_master="/tmp/master.npy",
        logger=logs.append,
        import_modules=lambda: (_ for _ in ()).throw(ImportError("missing dep")),
    )

    assert runtime.classifier is None
    assert any("required package not installed" in line for line in logs)
