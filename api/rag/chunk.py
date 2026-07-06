"""Split cleaned document text into retrievable chunks with stable metadata.

Consumes PageRecords from ingest.py and produces Chunk objects that carry the
full CHUNK DATA MODEL from the primer:

    chunk_id, text, source_doc, source_url, page/section, retrieved_on, embedding

source_url and retrieved_on are NOT invented here -- they come from a manifest
(corpus/manifest.json) that the human maintains, keyed by PDF file name. The
embedding field is left None; the embed step fills it later.

Chunking is word-windowed with a configurable size/overlap expressed in *tokens*
and converted to words via an English estimate (TOKENS_PER_WORD). This keeps
chunking deterministic and offline -- no tokenizer download, no API call.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

try:  # works when imported as a package (from .ingest ...)
    from .ingest import iter_corpus, CORPUS_RAW
except ImportError:  # works when run directly as `python rag/chunk.py`
    from ingest import iter_corpus, CORPUS_RAW

logger = logging.getLogger(__name__)

# corpus/manifest.json sits next to corpus/raw/.
MANIFEST_PATH = CORPUS_RAW.parent / "manifest.json"

# Defaults from the primer: ~500-token chunks with ~50-token overlap.
DEFAULT_CHUNK_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 50
# English averages ~1.3 subword tokens per whitespace word. Used only to size
# the word window and to *estimate* a chunk's token count for reporting.
TOKENS_PER_WORD = 1.3


@dataclass
class Chunk:
    """One retrieval unit with full provenance (the CHUNK DATA MODEL)."""

    chunk_id: str
    text: str
    source_doc: str
    source_url: str
    page: int
    retrieved_on: str
    section: Optional[str] = None
    embedding: Optional[list[float]] = None  # filled by the embed step

    def metadata(self) -> dict:
        """Flat metadata dict for the vector store (excludes text + embedding)."""
        return {
            "chunk_id": self.chunk_id,
            "source_doc": self.source_doc,
            "source_url": self.source_url,
            "page": self.page,
            "section": self.section or "",
            "retrieved_on": self.retrieved_on,
        }


@dataclass
class DocProvenance:
    source_url: str
    retrieved_on: str


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict[str, DocProvenance]:
    """Load {pdf_filename: {source_url, retrieved_on}} from the manifest JSON."""
    if not manifest_path.exists():
        logger.warning("No manifest at %s -- chunks will lack source_url/retrieved_on.",
                       manifest_path)
        return {}
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[str, DocProvenance] = {}
    for doc, meta in raw.items():
        out[doc] = DocProvenance(
            source_url=meta.get("source_url", ""),
            retrieved_on=meta.get("retrieved_on", ""),
        )
    return out


def _provenance_for(manifest: dict[str, DocProvenance], source_doc: str) -> DocProvenance:
    prov = manifest.get(source_doc)
    if prov is None:
        logger.warning("%s missing from manifest -- add source_url + retrieved_on.",
                       source_doc)
        return DocProvenance(source_url="", retrieved_on="")
    if not prov.source_url or not prov.retrieved_on:
        logger.warning("%s has incomplete manifest entry (url=%r, retrieved_on=%r).",
                       source_doc, prov.source_url, prov.retrieved_on)
    return prov


def _slug(source_doc: str) -> str:
    """Filename -> stable id stem: 'MUDRA Guidelines.pdf' -> 'mudra_guidelines'."""
    stem = Path(source_doc).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    return stem.strip("_")


def estimate_tokens(text: str) -> int:
    """Rough token count for a piece of text (reporting + sizing only)."""
    return round(len(text.split()) * TOKENS_PER_WORD)


def chunk_text(
    text: str,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[str]:
    """Sliding word-window over one page's text. Returns list of chunk strings."""
    words = text.split()
    if not words:
        return []

    words_per_chunk = max(1, round(chunk_tokens / TOKENS_PER_WORD))
    overlap_words = min(words_per_chunk - 1, round(overlap_tokens / TOKENS_PER_WORD))
    step = max(1, words_per_chunk - overlap_words)

    chunks: list[str] = []
    start = 0
    while start < len(words):
        window = words[start:start + words_per_chunk]
        chunks.append(" ".join(window))
        if start + words_per_chunk >= len(words):
            break  # this window reached the end; don't emit a duplicate tail
        start += step
    return chunks


def chunk_corpus(
    corpus_dir: Path = CORPUS_RAW,
    manifest_path: Path = MANIFEST_PATH,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> Iterator[Chunk]:
    """Yield Chunk objects for every PDF in the corpus, with provenance attached."""
    manifest = load_manifest(manifest_path)
    for rec in iter_corpus(corpus_dir):
        if not rec.text:
            continue  # empty page (ingest already warned)
        prov = _provenance_for(manifest, rec.source_doc)
        stem = _slug(rec.source_doc)
        for idx, ctext in enumerate(
            chunk_text(rec.text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
        ):
            yield Chunk(
                chunk_id=f"{stem}_p{rec.page_number}_c{idx}",
                text=ctext,
                source_doc=rec.source_doc,
                source_url=prov.source_url,
                page=rec.page_number,
                retrieved_on=prov.retrieved_on,
            )


def _summarise(corpus_dir: Path, manifest_path: Path,
               chunk_tokens: int, overlap_tokens: int) -> None:
    """Acceptance-check helper: counts, uniqueness, metadata + size stats."""
    chunks = list(chunk_corpus(corpus_dir, manifest_path,
                               chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens))
    print(f"Total chunks: {len(chunks)}  "
          f"(target ~{chunk_tokens} tokens, overlap ~{overlap_tokens})")
    if not chunks:
        print("No chunks produced. Add PDFs to corpus/raw/ and a manifest.json.")
        return

    # Per-document counts
    per_doc: dict[str, int] = {}
    for c in chunks:
        per_doc[c.source_doc] = per_doc.get(c.source_doc, 0) + 1
    print("\nChunks per document:")
    for doc, n in sorted(per_doc.items()):
        print(f"  {doc}: {n}")

    # Unique chunk_id check
    ids = [c.chunk_id for c in chunks]
    unique = len(set(ids)) == len(ids)
    print(f"\nUnique chunk_ids: {'PASS' if unique else 'FAIL'} "
          f"({len(set(ids))} unique / {len(ids)} total)")

    # Complete-metadata check (required provenance fields present)
    incomplete = [c.chunk_id for c in chunks
                  if not (c.source_doc and c.source_url and c.retrieved_on)]
    if incomplete:
        print(f"Metadata: {len(incomplete)} chunk(s) missing source_url/retrieved_on "
              f"-- fix manifest.json. e.g. {incomplete[:3]}")
    else:
        print("Metadata: PASS (all chunks have source_doc, source_url, retrieved_on)")

    # Size stats (estimated tokens)
    sizes = [estimate_tokens(c.text) for c in chunks]
    print(f"\nChunk size (est. tokens): min={min(sizes)} "
          f"mean={statistics.mean(sizes):.0f} max={max(sizes)}")

    # Sample chunk
    s = chunks[0]
    print(f"\nSample chunk:\n  chunk_id: {s.chunk_id}\n  page: {s.page}  "
          f"source: {s.source_doc}\n  url: {s.source_url}  retrieved_on: {s.retrieved_on}\n"
          f"  text: {s.text[:160]!r}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Chunk corpus PDFs with provenance metadata.")
    p.add_argument("--corpus", type=Path, default=CORPUS_RAW)
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    p.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    args = p.parse_args()
    _summarise(args.corpus, args.manifest, args.chunk_tokens, args.overlap_tokens)
