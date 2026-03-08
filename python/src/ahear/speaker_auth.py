"""Speaker verification primitives extracted from whispercpp-listen."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SpeakerAuthRuntime:
    classifier: Any | None = None
    voiceprint: Any | None = None
    np_module: Any | None = None
    torch_module: Any | None = None
    torchaudio_module: Any | None = None
    device: str = "cpu"


def speaker_auth_enabled(*, classifier: Any | None, voiceprint: Any | None) -> bool:
    return classifier is not None and voiceprint is not None


def initialize_speaker_auth_runtime(
    *,
    enabled: bool,
    requested_device: str,
    speaker_master: str,
    logger: Any,
    torchaudio_module: Any | None = None,
    torch_module: Any | None = None,
    np_module: Any | None = None,
    classifier_factory: Any | None = None,
    path_exists: Any = os.path.exists,
    load_voiceprint: Any | None = None,
    import_modules: Any | None = None,
) -> SpeakerAuthRuntime:
    runtime = SpeakerAuthRuntime(device=requested_device)
    if not enabled:
        return runtime

    logger(f"Speaker ID enabled: device={requested_device} master={speaker_master}")
    try:
        if import_modules is not None:
            torchaudio_module, torch_module, np_module, classifier_factory = import_modules()
        else:
            if torchaudio_module is None:
                import torchaudio as imported_torchaudio  # type: ignore[import-untyped]

                if not hasattr(imported_torchaudio, "list_audio_backends"):
                    imported_torchaudio.list_audio_backends = lambda: []
                torchaudio_module = imported_torchaudio
            if torch_module is None:
                import torch as imported_torch  # type: ignore[import-untyped]

                torch_module = imported_torch
            if np_module is None:
                import numpy as imported_np  # type: ignore[import-untyped]

                np_module = imported_np
            if classifier_factory is None:
                from speechbrain.inference.speaker import EncoderClassifier as _EC  # type: ignore[import-untyped]

                classifier_factory = _EC.from_hparams

        assert torchaudio_module is not None
        assert torch_module is not None
        assert np_module is not None
        assert classifier_factory is not None
        runtime.torchaudio_module = torchaudio_module
        runtime.torch_module = torch_module
        runtime.np_module = np_module
        device = requested_device
        if device == "cpu" and torch_module.cuda.is_available():
            device = "cuda:0"
            logger(f"Speaker ID: CUDA detected, upgrading device to {device}")
        runtime.device = device
        runtime.classifier = classifier_factory(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )

        loader = np_module.load if load_voiceprint is None else load_voiceprint
        if path_exists(speaker_master):
            runtime.voiceprint = loader(speaker_master)
            logger(f"Speaker ID: master voiceprint loaded from {speaker_master}")
        else:
            logger(f"Speaker ID warning: master voiceprint not found at {speaker_master}")
    except ImportError as exc:
        logger(
            f"Speaker ID error: required package not installed ({exc}). "
            "Install with: pip install speechbrain torchaudio torch numpy"
        )
    except Exception as exc:
        logger(f"Speaker ID initialization error: {exc}")

    return runtime


def verify_speaker_identity(
    *,
    wav_path: Path,
    classifier: Any | None,
    voiceprint: Any | None,
    torchaudio_module: Any,
    torch_module: Any,
    np_module: Any,
    device: str,
    threshold: float,
    topk: int,
    auth_error_text: str,
    logger: Any,
    log_label: str,
    intent: str,
) -> tuple[bool, str | None]:
    if not speaker_auth_enabled(classifier=classifier, voiceprint=voiceprint):
        return True, None
    assert classifier is not None
    assert voiceprint is not None

    try:
        started_at = time.time()
        signal, sample_rate = torchaudio_module.load(str(wav_path))
        if sample_rate != 16_000:
            resampler = torchaudio_module.transforms.Resample(sample_rate, 16_000)
            signal = resampler(signal)

        with torch_module.no_grad():
            embeddings = classifier.encode_batch(signal.to(device))
            embedding = embeddings.squeeze().cpu().numpy()
            if len(embedding.shape) > 1:
                embedding = np_module.mean(embedding, axis=0)
            embedding = embedding / np_module.linalg.norm(embedding)
            if len(voiceprint.shape) == 2:
                similarities = np_module.dot(voiceprint, embedding)
                top_similarities = sorted(similarities.tolist(), reverse=True)[:topk]
                similarity = float(np_module.mean(top_similarities))
            else:
                similarity = float(np_module.dot(voiceprint, embedding))

        elapsed = time.time() - started_at
        if similarity < threshold:
            logger(
                f"{log_label} AUTH FAILED: intent={intent} "
                f"similarity={similarity:.4f} (threshold={threshold}) SV_elapsed={elapsed:.2f}s"
            )
            return False, auth_error_text
        logger(
            f"{log_label} AUTH PASSED: intent={intent} "
            f"similarity={similarity:.4f} SV_elapsed={elapsed:.2f}s"
        )
        return True, None
    except Exception as exc:
        logger(f"Speaker ID verification error: {exc}")
        return False, auth_error_text
