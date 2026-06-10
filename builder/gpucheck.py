"""
builder/gpu_check.py
--------------------
GPU / provider detection for InsightFace model loading.
Single responsibility: figure out which ONNX execution provider to use.
Nothing else lives here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_providers() -> list[str]:
    """
    Return an ordered list of ONNX Runtime execution providers,
    best-first (GPU preferred, CPU fallback).

    Returns
    -------
    list[str]
        e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"]
        or   ["CPUExecutionProvider"]
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        logger.warning("onnxruntime not found; defaulting to CPU.")
        return ["CPUExecutionProvider"]

    preferred = [
        "CUDAExecutionProvider",       # NVIDIA GPU
        "DmlExecutionProvider",        # Windows DirectML (AMD/Intel GPU)
        "ROCMExecutionProvider",       # AMD ROCm (Linux)
        "CoreMLExecutionProvider",     # macOS/Apple Silicon
        "OpenVINOExecutionProvider",   # Intel iGPU/CPU via OpenVINO
    ]

    chosen = [p for p in preferred if p in available]
    chosen.append("CPUExecutionProvider")  # always last resort

    if len(chosen) > 1:
        logger.info("GPU provider selected: %s", chosen[0])
    else:
        logger.info("No GPU provider found; using CPU.")

    return chosen


def has_gpu() -> bool:
    """Return True if any non-CPU provider is available."""
    providers = get_providers()
    return any(p != "CPUExecutionProvider" for p in providers)


def provider_label() -> str:
    """
    Human-readable label for display in the UI.
    e.g. "CUDA (GPU)" or "CPU only"
    """
    providers = get_providers()
    first = providers[0]
    labels = {
        "CUDAExecutionProvider": "CUDA (NVIDIA GPU)",
        "DmlExecutionProvider": "DirectML (GPU)",
        "ROCMExecutionProvider": "ROCm (AMD GPU)",
        "CoreMLExecutionProvider": "CoreML (Apple GPU)",
        "OpenVINOExecutionProvider": "OpenVINO (Intel)",
        "CPUExecutionProvider": "CPU only",
    }
    return labels.get(first, first)