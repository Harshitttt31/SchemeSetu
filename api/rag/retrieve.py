"""Top-k retrieval over the Chroma store -- returns candidate chunks for a query.

retrieve(query, k) embeds the query with the SAME Gemini model used at ingestion
(task_type=RETRIEVAL_QUERY, the asymmetric partner to the RETRIEVAL_DOCUMENT
embeddings in the index), runs a cosine similarity search, and returns ranked
chunks with a similarity score and full provenance metadata.

The pure ranking step (search) is separated from embedding so it can be tested
offline without an API call.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Optional

try:  # package import
    from .embed import embed_query
    from . import store
except ImportError:  # direct `python rag/retrieve.py`
    from embed import embed_query
    import store

logger = logging.getLogger(__name__)

DEFAULT_K = 5


@dataclass
class RetrievedChunk:
    """One ranked search hit with score + provenance."""

    chunk_id: str
    text: str
    score: float          # cosine similarity in [-1, 1]; higher = more relevant
    distance: float       # raw Chroma cosine distance; lower = more relevant
    source_doc: str
    source_url: str
    page: object
    retrieved_on: str
    section: str


def search(collection, query_embedding: list[float], k: int = DEFAULT_K) -> list[RetrievedChunk]:
    """Run the similarity search against an open collection. No API call here."""
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma nests one list per query; we sent a single query -> index [0].
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    hits: list[RetrievedChunk] = []
    for cid, text, meta, dist in zip(ids, docs, metas, dists):
        hits.append(RetrievedChunk(
            chunk_id=cid,
            text=text,
            score=1.0 - dist,       # cosine distance -> similarity
            distance=dist,
            source_doc=meta.get("source_doc", ""),
            source_url=meta.get("source_url", ""),
            page=meta.get("page", ""),
            retrieved_on=meta.get("retrieved_on", ""),
            section=meta.get("section", ""),
        ))
    return hits


def retrieve(query: str, k: int = DEFAULT_K, *, client=None) -> list[RetrievedChunk]:
    """Embed the query and return the top-k chunks from the persisted index."""
    client = client or store.get_client()
    collection = store.get_collection(client)
    if collection.count() == 0:
        raise RuntimeError("Index is empty. Run scripts/build_index.py first.")
    query_embedding = embed_query(query)
    return search(collection, query_embedding, k)


def _print_hits(query: str, hits: list[RetrievedChunk]) -> None:
    print(f"\nQ: {query}")
    if not hits:
        print("  (no results)")
        return
    for rank, h in enumerate(hits, start=1):
        snippet = " ".join(h.text.split())[:160]
        print(f"  {rank}. score={h.score:.3f}  {h.chunk_id}  "
              f"[{h.source_doc} p{h.page}]")
        print(f"     {snippet!r}")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Query the SchemeSetu retrieval index.")
    p.add_argument("query", nargs="*", help="Question (omit for interactive mode).")
    p.add_argument("-k", type=int, default=DEFAULT_K, help="Number of chunks to return.")
    args = p.parse_args()

    if args.query:
        _print_hits(" ".join(args.query), retrieve(" ".join(args.query), args.k))
        return

    print("Interactive retrieval. Type a question (blank line or Ctrl-C to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        _print_hits(q, retrieve(q, args.k))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    _cli()
