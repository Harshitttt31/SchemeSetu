"""Chroma vector store: build, persist, and query the indexed corpus.

Chroma is used as pure storage: we pass in embeddings we computed ourselves
(no Chroma-side embedding function), plus the chunk text and metadata. The store
persists to api/data/index/ so the index survives across processes.

Cosine distance is configured to match the normalised Gemini embeddings.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import chromadb

if TYPE_CHECKING:  # avoid a runtime import cycle; store doesn't need chunk at runtime
    from .chunk import Chunk

logger = logging.getLogger(__name__)

INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "index"
COLLECTION_NAME = "schemesetu"


def get_client(persist_dir: Path = INDEX_DIR) -> chromadb.ClientAPI:
    """Return a persistent Chroma client rooted at persist_dir."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def build_collection(client: chromadb.ClientAPI, *, reset: bool = True):
    """Get the collection, optionally dropping it first for a clean rebuild."""
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # nothing to delete on a first build
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # match normalised embeddings
    )


def add_chunks(collection, chunks: "Iterable[Chunk]", embeddings: list[list[float]]) -> int:
    """Store chunks + their embeddings + metadata. Returns count added."""
    chunks = list(chunks)
    if len(chunks) != len(embeddings):
        raise ValueError(f"chunks ({len(chunks)}) != embeddings ({len(embeddings)})")
    if not chunks:
        return 0
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[c.metadata() for c in chunks],
    )
    return len(chunks)


def get_collection(client: chromadb.ClientAPI):
    """Open the existing collection for querying (retrieve step uses this)."""
    return client.get_collection(COLLECTION_NAME)
