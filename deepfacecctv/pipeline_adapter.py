"""Adapter between typed Config and existing pipeline.py at project root."""

from __future__ import annotations

import sys
from pathlib import Path

# ── CRITICAL: Add project root to path BEFORE any builder imports ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from deepfacecctv.config import Config


def run_pipeline(cfg: Config) -> None:
    """Start detection pipeline by calling pipeline.run() from project root."""
    legacy_cfg = cfg.to_legacy_dict()

    # Import pipeline.py from project root and call its run() function
    import pipeline as _pipeline_module

    _pipeline_module.run(legacy_cfg)


def start_dashboard(cfg: Config) -> None:
    """Start Flask dashboard independently."""
    try:
        from dashboard.app import create_app

        app = create_app(
            db_path=cfg.db_path,
            detections_db=cfg.output_db,
            host=cfg.flask_host,
            port=cfg.flask_port,
        )
        app.run(host=cfg.flask_host, port=cfg.flask_port, debug=False)
    except ImportError:
        from dashboard.app import app

        app.run(host=cfg.flask_host, port=cfg.flask_port, debug=False)


# ── Gallery management ─────────────────────────────────────────────────────


def _ensure_builder_importable():
    """Ensure builder/ is importable by adding project root to sys.path."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def list_gallery_identities(
    db_path: str,
    collection_name: str,
) -> list[tuple[str, int, str | None]]:
    """List all enrolled identities with embedding counts."""
    _ensure_builder_importable()

    from builder.db_ops import get_collection

    collection = get_collection(db_path=db_path, collection_name=collection_name)

    # Get all metadata
    results = collection.get(include=["metadatas"])

    if not results or not results.get("metadatas"):
        return []

    # Group by name
    from collections import defaultdict

    name_counts = defaultdict(int)
    name_dates = {}

    for meta in results["metadatas"]:
        if meta and "name" in meta:
            name = meta["name"]
            name_counts[name] += 1
            if "added_at" in meta:
                name_dates[name] = meta["added_at"]

    return [(name, count, name_dates.get(name)) for name, count in sorted(name_counts.items())]


def get_gallery_info(
    db_path: str,
    collection_name: str,
) -> dict:
    """Get gallery statistics."""
    _ensure_builder_importable()

    from builder.db_ops import get_collection

    collection = get_collection(db_path=db_path, collection_name=collection_name)
    results = collection.get(include=["metadatas"])

    identities = set()
    embeddings = 0

    if results and results.get("metadatas"):
        for meta in results["metadatas"]:
            if meta and "name" in meta:
                identities.add(meta["name"])
                embeddings += 1

    # Calculate DB size
    db_size = 0
    db_path_obj = Path(db_path)
    if db_path_obj.exists():
        db_size = sum(f.stat().st_size for f in db_path_obj.rglob("*") if f.is_file())

    return {
        "identities": len(identities),
        "embeddings": embeddings,
        "size": f"{db_size / (1024*1024):.1f} MB",
    }


def enroll_identity(
    name: str,
    image_paths: list[Path],
    db_path: str,
    collection_name: str,
    yunet_model: str,
) -> dict:
    """Enroll a new person into the gallery."""
    _ensure_builder_importable()

    from builder.embedder import extract_embedding_from_file
    from builder.db_ops import add_embeddings

    embeddings = []
    img_paths = []

    for img_path in image_paths:
        emb = extract_embedding_from_file(str(img_path))
        if emb is not None:
            embeddings.append(emb.tolist())
            img_paths.append(str(img_path))

    if embeddings:
        add_embeddings(
            name=name,
            embeddings=embeddings,
            img_paths=img_paths,
            db_path=db_path,
            collection_name=collection_name,
        )

    return {
        "embeddings": len(embeddings),
        "images": len(image_paths),
    }


def delete_identity(
    name: str,
    db_path: str,
    collection_name: str,
) -> bool:
    """Delete an identity from the gallery."""
    _ensure_builder_importable()

    from builder.db_ops import get_collection

    collection = get_collection(db_path=db_path, collection_name=collection_name)

    # Find all embeddings for this name
    results = collection.get(where={"name": name}, include=["ids"])

    if not results or not results.get("ids"):
        return False

    # Delete by IDs
    collection.delete(ids=results["ids"])
    return True


# ── Dataset builder ────────────────────────────────────────────────────────


def build_dataset(
    input_dir: Path,
    output_dir: Path,
    db_path: str,
    collection_name: str,
    yunet_model: str,
    min_face_size: int,
    blur_threshold: float,
) -> dict:
    """Build dataset from organized images."""
    _ensure_builder_importable()

    from builder.embedder import extract_embedding_from_file
    from builder.db_ops import add_embeddings

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"identities": 0, "images": 0, "valid_faces": 0, "rejected": 0}

    for person_dir in sorted(input_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        name = person_dir.name
        embeddings = []
        img_paths = []

        for img_file in person_dir.iterdir():
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue

            stats["images"] += 1

            emb = extract_embedding_from_file(str(img_file))
            if emb is not None:
                embeddings.append(emb.tolist())
                img_paths.append(str(img_file))
                stats["valid_faces"] += 1
            else:
                stats["rejected"] += 1

        if embeddings:
            add_embeddings(
                name=name,
                embeddings=embeddings,
                img_paths=img_paths,
                db_path=db_path,
                collection_name=collection_name,
            )
            stats["identities"] += 1

    return stats


def update_dataset(
    new_images_dir: Path,
    existing_dataset: Path,
    db_path: str,
    collection_name: str,
    yunet_model: str,
) -> dict:
    """Update dataset with new images."""
    _ensure_builder_importable()

    from builder.embedder import extract_embedding_from_file
    from builder.db_ops import add_embeddings

    stats = {"new": 0, "updated": 0, "added": 0}

    for person_dir in sorted(new_images_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        name = person_dir.name
        embeddings = []
        img_paths = []

        for img_file in person_dir.iterdir():
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue

            emb = extract_embedding_from_file(str(img_file))
            if emb is not None:
                embeddings.append(emb.tolist())
                img_paths.append(str(img_file))

        if embeddings:
            add_embeddings(
                name=name,
                embeddings=embeddings,
                img_paths=img_paths,
                db_path=db_path,
                collection_name=collection_name,
            )
            stats["added"] += len(embeddings)
            stats["updated" if (existing_dataset / name).exists() else "new"] += 1

    return stats


def update_gallery(
    new_images_dir: Path,
    db_path: str,
    collection_name: str,
    yunet_model: str,
) -> dict:
    """Update gallery directly with new images."""
    _ensure_builder_importable()

    from builder.embedder import extract_embedding_from_file
    from builder.db_ops import add_embeddings

    stats = {"new": 0, "updated": 0, "added": 0}

    for person_dir in sorted(new_images_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        name = person_dir.name
        embeddings = []
        img_paths = []

        for img_file in person_dir.iterdir():
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue

            emb = extract_embedding_from_file(str(img_file))
            if emb is not None:
                embeddings.append(emb.tolist())
                img_paths.append(str(img_file))

        if embeddings:
            add_embeddings(
                name=name,
                embeddings=embeddings,
                img_paths=img_paths,
                db_path=db_path,
                collection_name=collection_name,
            )
            stats["added"] += len(embeddings)
            stats["new" if len(embeddings) == len(img_paths) else "updated"] += 1

    return stats


def update_gallery(
    new_images_dir: Path,
    db_path: str,
    collection_name: str,
    yunet_model: str,
) -> dict:
    """Update gallery directly with new images."""
    _ensure_builder_importable()

    from builder.embedder import extract_embeddings
    from builder.db_ops import add_embeddings

    stats = {"new": 0, "updated": 0, "added": 0}

    for person_dir in sorted(new_images_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        name = person_dir.name
        embeddings = []
        img_paths = []

        for img_file in person_dir.iterdir():
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue

            emb = extract_embeddings(str(img_file), yunet_model)
            if emb is not None:
                embeddings.append(emb.tolist())
                img_paths.append(str(img_file))

        if embeddings:
            add_embeddings(
                name=name,
                embeddings=embeddings,
                img_paths=img_paths,
                db_path=db_path,
                collection_name=collection_name,
            )
            stats["added"] += len(embeddings)
            stats["new" if len(embeddings) == len(img_paths) else "updated"] += 1

    return stats


def get_dataset_info(dataset_dir: Path) -> dict:
    """Get dataset statistics."""
    identities = 0
    total_images = 0

    for person_dir in sorted(dataset_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        identities += 1
        total_images += sum(
            1
            for f in person_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

    size_mb = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file()) / (1024 * 1024)

    return {
        "identities": identities,
        "total_images": total_images,
        "avg_images": round(total_images / max(identities, 1), 1),
        "size_mb": f"{size_mb:.1f} MB",
    }
