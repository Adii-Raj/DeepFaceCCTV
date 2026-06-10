"""
builder/embedder.py
-------------------
Embedding extraction logic for building the face dataset.
Single responsibility: take an image (or folder of images), detect a face,
return a 512-dim ArcFace embedding.
No Tkinter, no ChromaDB, no pipeline logic.

Depends on:
    builder/gpu_check.py  — provider selection
    core/detector.py      — YuNet face detection (reused from pipeline)
    insightface / onnxruntime
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Supported image extensions
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ArcFace buffalo_l input: 112×112 BGR
_ARC_SIZE = (112, 112)

# Default model path (relative to project root)
_DEFAULT_MODEL_DIR = "models/buffalo_l"
_ARCFACE_ONNX = Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"

# GPU Result codes
GPU_OK = "ok"
GPU_NO_PACKAGE = "no_package"
GPU_NO_CUDA = "no_cuda"
GPU_NO_DEVICE = "no_device"
GPU_UNKNOWN_ERROR = "unknown_error"

GPU_MESSAGES = {
    GPU_OK: "✅ GPU ready — CUDA is available",
    GPU_NO_PACKAGE: "❌ onnxruntime-gpu is not installed\n"
                    "    Fix: pip uninstall onnxruntime\n"
                    "         pip install onnxruntime-gpu",
    GPU_NO_CUDA: "❌ CUDA drivers / toolkit not found on this PC\n"
                 "    Fix: install NVIDIA CUDA Toolkit from nvidia.com",
    GPU_NO_DEVICE: "❌ No compatible NVIDIA GPU detected\n"
                   "    This PC may not have a CUDA-capable graphics card.",
    GPU_UNKNOWN_ERROR: "❌ Unknown error during GPU check — see log for details",
}


# ---------------------------------------------------------------------------
# GPU Detection
# ---------------------------------------------------------------------------

def check_gpu_support() -> tuple[str, str]:
    """
    Run three checks in order:
      1. Is onnxruntime-gpu installed?
      2. Is CUDAExecutionProvider listed as available?
      3. Can we actually create a session on CUDA without error?

    Returns (result_code, detail_message).
    Safe to call from a background thread.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return GPU_NO_PACKAGE, "onnxruntime is not installed at all"

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        return GPU_NO_CUDA, f"Available providers: {available}"

    try:
        if _ARCFACE_ONNX.exists():
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            ort.InferenceSession(
                str(_ARCFACE_ONNX),
                sess_options=sess_options,
                providers=["CUDAExecutionProvider"],
            )
        return GPU_OK, "CUDAExecutionProvider verified"
    except Exception as e:
        err = str(e).lower()
        if "cudnn" in err or "cuda" in err or "driver" in err:
            return GPU_NO_CUDA, str(e)
        if "no devices" in err or "device" in err:
            return GPU_NO_DEVICE, str(e)
        return GPU_UNKNOWN_ERROR, str(e)


def get_providers(use_gpu: bool = False) -> list[str]:
    """Return appropriate execution providers based on GPU availability."""
    if use_gpu:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class ArcFaceEmbedder:
    """
    Wraps InsightFace's ArcFace model for embedding extraction.
    One instance per session — loading is expensive (~1–2 s).
    """

    def __init__(self, model_dir: str = _DEFAULT_MODEL_DIR, use_gpu: bool = False):
        self._model_dir = Path(model_dir)
        self._model = None
        self._use_gpu = use_gpu
        self._providers = get_providers(use_gpu)
        logger.info("ArcFaceEmbedder initialising with providers: %s", self._providers)

    def _load(self):
        """Lazy-load InsightFace model on first use."""
        if self._model is not None:
            return

        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "insightface is not installed. Run: pip install insightface"
            ) from e

        ctx_id = 0 if self._use_gpu else -1
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(self._model_dir.parent),
            providers=self._providers,
        )
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self._model = app
        logger.info("InsightFace model loaded from %s (GPU=%s)", self._model_dir, self._use_gpu)

    def embed(self, bgr_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-dim ArcFace embedding from a BGR image.

        Parameters
        ----------
        bgr_image : np.ndarray
            BGR image (any size). Must contain exactly one clear face.

        Returns
        -------
        np.ndarray of shape (512,), L2-normalised,
        or None if no face is detected.
        """
        self._load()
        faces = self._model.get(bgr_image)

        if not faces:
            logger.debug("embed(): no face detected in image")
            return None

        if len(faces) > 1:
            logger.debug("embed(): %d faces found; using largest", len(faces))
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )

        emb = faces[0].normed_embedding
        return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def load_images_from_folder(folder: str | Path) -> list[tuple[Path, np.ndarray]]:
    """
    Load all supported images from a folder (non-recursive).

    Returns
    -------
    list of (Path, np.ndarray) tuples, BGR images.
    Empty list if folder is empty or has no supported images.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    images = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in _IMG_EXTS:
            img = cv2.imread(str(p))
            if img is None:
                logger.warning("Could not read image: %s", p)
                continue
            images.append((p, img))

    logger.info("Loaded %d image(s) from %s", len(images), folder)
    return images


def extract_embeddings_from_folder(
    folder: str | Path,
    embedder: ArcFaceEmbedder,
    max_images: int = 0,
    progress_callback=None,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """
    Extract embeddings for all images in a folder.

    Parameters
    ----------
    folder            : path to image folder
    embedder          : ArcFaceEmbedder instance
    max_images        : max images to process (0 = all)
    progress_callback : optional callable(current: int, total: int, path: str)
                        called after each image for UI progress updates

    Returns
    -------
    (embeddings, img_paths, failed_paths)
        embeddings  : list of (512,) float32 arrays
        img_paths   : parallel list of source image path strings
        failed_paths: images where no face was detected
    """
    images = load_images_from_folder(folder)
    if max_images > 0:
        images = images[:max_images]
    
    total = len(images)

    embeddings: list[np.ndarray] = []
    img_paths: list[str] = []
    failed_paths: list[str] = []

    for i, (path, img) in enumerate(images):
        emb = embedder.embed(img)
        if emb is not None:
            embeddings.append(emb)
            img_paths.append(str(path))
        else:
            failed_paths.append(str(path))
            logger.warning("No face found in: %s", path)

        if progress_callback:
            progress_callback(i + 1, total, str(path))

    logger.info(
        "Extracted %d/%d embeddings from '%s' (%d failed)",
        len(embeddings), total, folder, len(failed_paths),
    )
    return embeddings, img_paths, failed_paths


def extract_embedding_from_file(
    image_path: str | Path,
    embedder: ArcFaceEmbedder,
) -> Optional[np.ndarray]:
    """
    Extract a single embedding from one image file.

    Returns
    -------
    np.ndarray (512,) or None if no face detected / file unreadable.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error("Cannot read file: %s", image_path)
        return None
    return embedder.embed(img)