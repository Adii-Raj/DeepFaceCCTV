"""
builder/db_ops.py
-----------------
ChromaDB operations for the face gallery.
Single responsibility: create, read, update, delete people + their embeddings.
No Tkinter, no model loading, no detection logic.

Collection schema
-----------------
Each document in the ChromaDB collection represents ONE embedding vector.
  id        : "<name>_<uuid4>"          unique per embedding
  embedding : list[float] (512-dim)     ArcFace embedding
  metadata  :
    name    : str                       person's display name
    img_path: str                       source image path (for audit)
    added_at: str                       ISO-8601 timestamp
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "data/face_db"
_DEFAULT_COLLECTION = "face_gallery"


# ---------------------------------------------------------------------------
# Client / collection helpers
# ---------------------------------------------------------------------------

def _get_client(db_path: str = _DEFAULT_DB_PATH):
    """Return a persistent ChromaDB client."""
    try:
        import chromadb
    except ImportError as e:
        raise ImportError("chromadb is not installed. Run: pip install chromadb") from e

    Path(db_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)


def get_collection(
    db_path: str = _DEFAULT_DB_PATH,
    collection_name: str = _DEFAULT_COLLECTION,
):
    """
    Return (or create) the ChromaDB collection.
    Uses cosine distance — correct for normalised ArcFace embeddings.
    """
    client = _get_client(db_path)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def list_all_collections(db_path: str = _DEFAULT_DB_PATH) -> list[str]:
    """List all available ChromaDB collections."""
    try:
        client = _get_client(db_path)
        return sorted([c.name for c in client.list_collections()])
    except Exception as e:
        logger.error("Failed to list collections: %s", e)
        return []


def delete_entire_collection(
    collection_name: str,
    db_path: str = _DEFAULT_DB_PATH,
):
    """Permanently delete an entire ChromaDB collection."""
    try:
        client = _get_client(db_path)
        client.delete_collection(collection_name)
        logger.info("Deleted entire collection: '%s'", collection_name)
    except Exception as e:
        logger.error("Failed to delete collection '%s': %s", collection_name, e)
        raise


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def add_embeddings(
    name: str,
    embeddings: list[list[float]],
    img_paths: list[str],
    db_path: str = _DEFAULT_DB_PATH,
    collection_name: str = _DEFAULT_COLLECTION,
) -> int:
    """
    Insert one or more embeddings for a person.

    Parameters
    ----------
    name        : display name (e.g. "Aditya Kumar")
    embeddings  : list of 512-dim float lists
    img_paths   : parallel list of source image paths (same length)
    db_path     : ChromaDB storage directory
    collection_name

    Returns
    -------
    int : number of embeddings inserted
    """
    if len(embeddings) != len(img_paths):
        raise ValueError("embeddings and img_paths must have the same length")

    if not embeddings:
        logger.warning("add_embeddings called with empty list for '%s'", name)
        return 0

    col = get_collection(db_path, collection_name)
    ts = datetime.now(timezone.utc).isoformat()

    ids = [f"{name}_{uuid.uuid4().hex[:8]}" for _ in embeddings]
    metadatas = [
        {"name": name, "img_path": str(p), "added_at": ts}
        for p in img_paths
    ]

    col.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    logger.info("Added %d embedding(s) for '%s'", len(embeddings), name)
    return len(embeddings)


def delete_person(
    name: str,
    db_path: str = _DEFAULT_DB_PATH,
    collection_name: str = _DEFAULT_COLLECTION,
) -> int:
    """
    Delete ALL embeddings for a person by name.

    Returns
    -------
    int : number of embeddings deleted
    """
    col = get_collection(db_path, collection_name)
    results = col.get(where={"name": name})
    ids = results.get("ids", [])

    if not ids:
        logger.warning("delete_person: no embeddings found for '%s'", name)
        return 0

    col.delete(ids=ids)
    logger.info("Deleted %d embedding(s) for '%s'", len(ids), name)
    return len(ids)


def delete_multiple_people(
    names: list[str],
    db_path: str = _DEFAULT_DB_PATH,
    collection_name: str = _DEFAULT_COLLECTION,
) -> int:
    """
    Delete embeddings for multiple people at once.

    Returns
    -------
    int : total number of embeddings deleted
    """
    total_deleted = 0
    for name in names:
        deleted = delete_person(name, db_path, collection_name)
        total_deleted += deleted
    return total_deleted


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def list_people(
    db_path: str = _DEFAULT_DB_PATH,
    collection_name: str = _DEFAULT_COLLECTION,
) -> list[dict]:
    """
    Return a list of unique people with embedding counts.

    Returns
    -------
    list of dicts:
        [{"name": "Alice", "count": 5, "added_at": "2025-..."}, ...]
    sorted alphabetically by name.
    """
    col = get_collection(db_path, collection_name)
    results = col.get(include=["metadatas"])
    metadatas = results.get("metadatas") or []

    summary: dict[str, dict] = {}
    for meta in metadatas:
        n = meta.get("name", "unknown")
        if n not in summary:
            summary[n] = {"name": n, "count": 0, "added_at": meta.get("added_at", "")}
        summary[n]["count"] += 1

    return sorted(summary.values(), key=lambda x: x["name"].lower())


def count_embeddings(
    name: Optional[str] = None,
    db_path: str = _DEFAULT_DB_PATH,
    collection_name: str = _DEFAULT_COLLECTION,
) -> int:
    """
    Count total embeddings, or embeddings for a specific person.
    """
    col = get_collection(db_path, collection_name)
    if name:
        results = col.get(where={"name": name})
        return len(results.get("ids", []))
    return col.count()


def person_exists(
    name: str,
    db_path: str = _DEFAULT_DB_PATH,
    collection_name: str = _DEFAULT_COLLECTION,
) -> bool:
    """Return True if the person has at least one embedding in the DB."""
    return count_embeddings(name, db_path, collection_name) > 0


def collection_stats(
    db_path: str = _DEFAULT_DB_PATH,
    collection_name: str = _DEFAULT_COLLECTION,
) -> dict:
    """
    Return a summary dict useful for showing in the UI.

    Returns
    -------
    {"total_embeddings": int, "total_people": int, "people": list[dict]}
    """
    people = list_people(db_path, collection_name)
    return {
        "total_embeddings": sum(p["count"] for p in people),
        "total_people": len(people),
        "people": people,
    }