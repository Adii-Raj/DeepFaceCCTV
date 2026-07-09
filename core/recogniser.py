"""
core/recogniser.py
──────────────────
Face recognition — embedding extraction only.

Single responsibility: given a face crop (BGR), return an L2-normalised
embedding vector (np.ndarray float32).

Two backends:
  - InsightFaceRecogniser  : ArcFace buffalo_l (512-dim). Preferred.
  - SFaceRecogniser        : OpenCV SFace (128-dim). Fallback if insightface
                             is not installed.

To swap to a different model in the future:
  - Add a new class here implementing embed() with the same signature
  - Change pipeline.py to instantiate the new class
  - Nothing else changes
"""

import os
import socket
import urllib.request

import cv2
import numpy as np

# ── SFace model info ──────────────────────────────────────────────────────────

SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_MODEL_FILENAME = "face_recognition_sface_2021dec.onnx"
DOWNLOAD_TIMEOUT = 30

ARCFACE_MODEL_REL = "buffalo_l/w600k_r50.onnx"


# ── Model download helper ─────────────────────────────────────────────────────

def _ensure_model(url: str, path: str) -> str:
    if os.path.exists(path):
        return path
    print(f"[recogniser] Downloading {os.path.basename(path)} ...")
    try:
        def _hook(count, block, total):
            if total > 0 and count % 50 == 0:
                print(f"\r  {min(100, int(count*block*100/total))}%",
                      end="", flush=True)
        old = socket.getdefaulttimeout()
        socket.setdefaulttimeout(DOWNLOAD_TIMEOUT)
        try:
            urllib.request.urlretrieve(url, path, _hook)
        finally:
            socket.setdefaulttimeout(old)
        print(f"\n[recogniser] Saved -> {path}")
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        raise RuntimeError(f"[recogniser] Download failed: {e}") from e
    return path


# ── Normalisation helper ──────────────────────────────────────────────────────

def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-10 else vec


# ── InsightFace ArcFace ───────────────────────────────────────────────────────

class InsightFaceRecogniser:
    """
    ArcFace buffalo_l via insightface model_zoo.
    Produces 512-dim L2-normalised embeddings.

    Requires: pip install insightface onnxruntime
    """

    EMBEDDING_DIM = 512

    def __init__(self, ctx_id: int = 0, model_dir : str = None):
        print("[recogniser] Loading InsightFace ArcFace (w600k_r50) ...")
        try:
            import insightface.model_zoo as model_zoo
        except ImportError:
            raise RuntimeError(
                "insightface is not installed.\n"
                "Run: pip install insightface onnxruntime"
            )

        if model_dir:
            home = os.path.expanduser(model_dir)
        else:
            home = os.path.expanduser("~/.insightface/models")
        
        onnx_path = os.path.join(home, ARCFACE_MODEL_REL)

        if not os.path.exists(onnx_path):
            print("[recogniser] buffalo_l not cached — downloading (~330 MB) ...")
            self._trigger_insightface_download()

        if not os.path.exists(onnx_path):
            raise RuntimeError(
                f"[recogniser] ArcFace model not found at {onnx_path}.\n"
                "Check internet connection and try again."
            )

        self._rec = model_zoo.get_model(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        self._rec.prepare(ctx_id=ctx_id)
        print(f"[recogniser] ArcFace ready ({onnx_path})")

    @staticmethod
    def _trigger_insightface_download():
        """Use FaceAnalysis prepare() to trigger buffalo_l download."""
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l",
                               providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(160, 160))
            del app
        except Exception:
            pass

    @staticmethod
    def is_available() -> bool:
        """Returns True if insightface package is installed."""
        try:
            import insightface  # noqa: F401
            return True
        except ImportError:
            return False

    def embed(self, bgr_crop: np.ndarray) -> np.ndarray:
        """
        Embed a BGR face crop.
        Resizes to 112x112, converts to RGB, returns 512-dim float32 vector.
        Returns zero vector on failure.
        """
        try:
            resized = cv2.resize(bgr_crop, (112, 112))
            rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            feat    = self._rec.get_feat(rgb)
            feat    = np.array(feat).flatten().astype(np.float32)
            return _l2_normalise(feat)
        except Exception as e:
            print(f"[recogniser] ArcFace embed failed: {e}")
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)


# ── SFace fallback ────────────────────────────────────────────────────────────

class SFaceRecogniser:
    """
    OpenCV SFace recogniser.
    Produces 128-dim L2-normalised embeddings.

    Used as fallback when insightface is not installed.
    Note: embeddings are NOT compatible with InsightFace ArcFace embeddings.
          Do not mix SFace and ArcFace databases.
    """

    EMBEDDING_DIM = 128

    def __init__(self, model_path: str):
        _ensure_model(SFACE_MODEL_URL, model_path)
        self._rec = cv2.FaceRecognizerSF.create(model_path, "")
        print(f"[recogniser] SFace ready ({model_path})")

    def embed(self, bgr_crop: np.ndarray) -> np.ndarray:
        """
        Embed a BGR face crop (no landmark alignment — crop-only path).
        Returns 128-dim float32 vector. Returns zero vector on failure.
        """
        try:
            resized = cv2.resize(bgr_crop, (112, 112))
            feat    = self._rec.feature(resized)
            feat    = np.array(feat).flatten().astype(np.float32)
            return _l2_normalise(feat)
        except Exception as e:
            print(f"[recogniser] SFace embed failed: {e}")
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

    def embed_with_landmarks(
        self,
        bgr_full_frame: np.ndarray,
        face_row: np.ndarray,
    ) -> np.ndarray:
        """
        Embed using landmark-aligned crop (more accurate than crop-only).
        Requires the full frame and the YuNet face_row for alignment.
        Returns 128-dim float32 vector.
        """
        try:
            aligned = self._rec.alignCrop(
                bgr_full_frame, face_row.reshape(1, -1)
            )
            feat = self._rec.feature(aligned)[0].astype(np.float32)
            return _l2_normalise(feat)
        except Exception as e:
            print(f"[recogniser] SFace landmark embed failed: {e}")
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)


# ── Factory ───────────────────────────────────────────────────────────────────

def load_recogniser(sface_model_path: str = SFACE_MODEL_FILENAME, model_dir: str = None,):
    """
    Returns the best available recogniser.
    Prefers InsightFace ArcFace; falls back to SFace if not installed.
    """
    if InsightFaceRecogniser.is_available():
        print("[recogniser] insightface found — using ArcFace buffalo_l")
        return InsightFaceRecogniser(model_dir = model_dir)
    else:
        print("[recogniser] insightface not found — falling back to SFace")
        return SFaceRecogniser(sface_model_path)