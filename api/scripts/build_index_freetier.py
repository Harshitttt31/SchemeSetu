r"""Free-tier-safe index builder: embed ONE chunk per request, well spaced.

Why this exists (build_index.py stays the canonical builder):
    On the Gemini FREE tier, gemini-embedding-001 meters embedding so tightly that
    a multi-chunk batchEmbedContents call 429s (RESOURCE_EXHAUSTED) even after a
    full minute of cooldown, while a single-content request succeeds. The default
    batch_size=32 in build_index.py therefore cannot complete on free tier, and
    embed.py's burst-retry backoff is self-defeating (it fires 6 requests into the
    same rate-limited minute). This script sidesteps both by sending one chunk at a
    time with a fixed gap, so every request stays under the per-minute ceiling.

It is RESUMABLE: each embedding is appended to data/emb_cache.jsonl as it succeeds,
so a restart skips already-embedded chunks. When every chunk is cached, it rebuilds
the Chroma collection from the cache (same store.add_chunks path as the real build).

Reuses embed.py's EMBED_MODEL + RETRIEVAL_DOCUMENT so the vectors are identical to
what the canonical pipeline would produce. No frozen module is modified.

Run (background recommended, ~15-20 min on free tier):
    .\.venv\Scripts\python.exe scripts\build_index_freetier.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

import httpx
from google.genai import errors, types
from rag.chunk import chunk_corpus
from rag.embed import EMBED_MODEL, TASK_DOCUMENT, get_client
from rag import store

CACHE_PATH = store.INDEX_DIR.parent / "emb_cache.jsonl"   # api/data/emb_cache.jsonl
GAP = 13.0            # seconds between requests -> ~4.6/min, under the free-tier ceiling
COOLDOWN = 30.0      # on a 429, wait a fresh window before retrying the SAME chunk
NET_RETRY = 5.0      # on a transient network/DNS blip, brief pause then retry
MAX_ATTEMPTS = 8     # per-chunk attempts before we give up (surfaces a daily-cap wall)


def load_cache() -> dict[str, list[float]]:
    """Return {chunk_id: embedding} already computed in a prior/partial run."""
    cache: dict[str, list[float]] = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["chunk_id"]] = rec["embedding"]
    return cache


def append_cache(chunk_id: str, embedding: list[float]) -> None:
    """Persist one embedding immediately so progress survives an interruption."""
    with CACHE_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"chunk_id": chunk_id, "embedding": embedding}) + "\n")


def embed_one(client, text: str) -> list[float]:
    """Embed a single chunk, self-healing on 429 with a full-window cooldown."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL, contents=[text],
                config=types.EmbedContentConfig(task_type=TASK_DOCUMENT))
            return resp.embeddings[0].values
        except errors.APIError as exc:
            if getattr(exc, "code", None) == 429 and attempt < MAX_ATTEMPTS:
                print(f"    429 (attempt {attempt}); cooldown {COOLDOWN}s ...", flush=True)
                time.sleep(COOLDOWN)
                continue
            raise
        except httpx.TransportError as exc:  # DNS/connect/read blip -> transient
            if attempt < MAX_ATTEMPTS:
                print(f"    network error ({exc!r}, attempt {attempt}); "
                      f"retry in {NET_RETRY}s ...", flush=True)
                time.sleep(NET_RETRY)
                continue
            raise
    raise RuntimeError("unreachable")


def main() -> None:
    chunks = list(chunk_corpus())
    cache = load_cache()
    todo = [c for c in chunks if c.chunk_id not in cache]
    print(f"chunks={len(chunks)} cached={len(cache)} todo={len(todo)} "
          f"| model={EMBED_MODEL} task={TASK_DOCUMENT} GAP={GAP}s", flush=True)

    client = get_client()
    for n, c in enumerate(todo, start=1):
        emb = embed_one(client, c.text)
        append_cache(c.chunk_id, emb)
        cache[c.chunk_id] = emb
        print(f"  [{n}/{len(todo)}] embedded {c.chunk_id}", flush=True)
        if n < len(todo):
            time.sleep(GAP)

    missing = [c.chunk_id for c in chunks if c.chunk_id not in cache]
    if missing:
        raise SystemExit(f"Incomplete: {len(missing)} chunks still unembedded: {missing[:3]}")

    print("all chunks embedded; rebuilding Chroma collection ...", flush=True)
    coll = store.build_collection(store.get_client(), reset=True)
    embeddings = [cache[c.chunk_id] for c in chunks]   # aligned to chunk order
    store.add_chunks(coll, chunks, embeddings)
    print(f"\nDone. Indexed {coll.count()} chunks into {store.INDEX_DIR}", flush=True)


if __name__ == "__main__":
    main()
