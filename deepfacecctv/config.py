"""Typed configuration management."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class Config(BaseModel):
    """Pipeline configuration with sensible defaults."""

    source: Optional[str] = Field(
        default=None, description="Video source: file path, camera index (0), or RTSP/HTTP URL"
    )

    db_path: str = "data/face_db"
    collection_name: str = "face_gallery"

    yunet_model: str = "models/face_detection_yunet_2023mar.onnx"
    sface_model: str = "models/face_recognition_sface_2021dec.onnx"

    det_confidence: float = 0.55
    det_scale: float = 0.75

    threshold_accept: float = 0.48
    threshold_reject: float = 0.32

    max_yaw: float = 45.0
    min_face_size: int = 25
    blur_threshold: float = 20.0

    headless: bool = True
    recognition_interval: int = 4
    log_cooldown: float = 3.0
    gallery_refresh_sec: int = 60

    output_db: str = "data/detections.db"
    crops_dir: str = "dashboard/static/crops"

    flask_port: int = 5002
    flask_host: str = "0.0.0.0"

    transport: str = "tcp"

    @property
    def is_rtsp(self) -> bool:
        if self.source is None:
            return False
        return isinstance(self.source, str) and self.source.lower().startswith("rtsp://")

    @property
    def is_http_stream(self) -> bool:
        if self.source is None:
            return False
        return isinstance(self.source, str) and self.source.lower().startswith(
            ("http://", "https://")
        )

    @property
    def is_camera(self) -> bool:
        if self.source is None:
            return False
        return str(self.source) in ("0", "1", "2", "3", "4")

    @property
    def source_label(self) -> str:
        if self.source is None:
            return "none"
        if self.is_rtsp:
            return f"RTSP: {self.source}"
        if self.is_http_stream:
            return f"HTTP: {self.source}"
        if self.is_camera:
            return f"Camera {self.source}"
        return f"Video: {self.source}"

    @field_validator("source", mode="before")
    @classmethod
    def _coerce_source(cls, v: Any) -> Optional[str]:
        if v is None or v == "":
            return None
        if isinstance(v, int):
            return str(v)
        return str(v).strip() or None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert to legacy format for existing pipeline.py."""
        return {
            "rtsp": self.source if self.is_rtsp else None,
            "video": None if (self.is_rtsp or self.is_http_stream) else self.source,
            "transport": self.transport,
            "headless": self.headless,
            "db_path": self.db_path,
            "collection_name": self.collection_name,
            "refresh_interval": self.gallery_refresh_sec,
            "output_db": self.output_db,
            "yunet_model": self.yunet_model,
            "sface_model": self.sface_model,
            "threshold_accept": self.threshold_accept,
            "threshold_reject": self.threshold_reject,
            "border_margin": 0.05,
            "max_yaw": self.max_yaw,
            "min_face_size": self.min_face_size,
            "blur_threshold": self.blur_threshold,
            "det_scale": self.det_scale,
            "det_confidence": self.det_confidence,
            "recognition_interval": self.recognition_interval,
            "log_cooldown": self.log_cooldown,
        }


def load_config(path: Path | str = "config.json") -> Config:
    path = Path(path)
    if not path.exists():
        return Config()

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Migrate old keys
    migrations = {
        "video": "source",
        "rtsp_url": "source",
        "detections_db": "output_db",
        "refresh_interval": "gallery_refresh_sec",
    }
    for old, new in migrations.items():
        if old in raw and new not in raw:
            raw[new] = raw.pop(old)
        elif old in raw:
            raw.pop(old)

    return Config(**raw)


def save_config(cfg: Config, path: Path | str = "config.json") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
        f.write("\n")
