"""One-command pipeline: ingest -> chunk -> embed -> store into persistent Chroma.

Run from the api/ directory:
    python scripts/build_index.py

Requires GEMINI_API_KEY in api/.env and PDFs in api/corpus/raw/ (+ manifest.json).
Rebuilds the collection from scratch each run, so it is reproducible.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the `rag` package importable when run as a plain script.
API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from rag import store
from rag.chunk import chunk_corpus
from rag.embed import EMBED_MODEL, TASK_DOCUMENT, embed_texts

logger = logging.getLogger(__name__)


def build_index(corpus_dir: Path, manifest_path: Path, batch_size: int) -> int:
    """Run the full pipeline and return the number of chunks indexed."""
    print("1/4 ingest + chunk ...")
    chunks = list(chunk_corpus(corpus_dir, manifest_path))
    if not chunks:
        print("No chunks produced. Add PDFs to corpus/raw/ and entries to manifest.json.")
        return 0
    print(f"    {len(chunks)} chunks from the corpus.")

    print(f"2/4 embed with {EMBED_MODEL} (task={TASK_DOCUMENT}) ...")
    embeddings = embed_texts([c.text for c in chunks],
                             task_type=TASK_DOCUMENT, batch_size=batch_size)

    print("3/4 (re)build Chroma collection ...")
    client = store.get_client()
    collection = store.build_collection(client, reset=True)

    print("4/4 store embeddings + text + metadata ...")
    store.add_chunks(collection, chunks, embeddings)

    count = collection.count()
    print(f"\nDone. Indexed {count} chunks into {store.INDEX_DIR}")
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from rag.chunk import CORPUS_RAW, MANIFEST_PATH
    from rag.embed import DEFAULT_BATCH_SIZE

    p = argparse.ArgumentParser(description="Build the SchemeSetu Chroma index.")
    p.add_argument("--corpus", type=Path, default=CORPUS_RAW)
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = p.parse_args()
    build_index(args.corpus, args.manifest, args.batch_size)
