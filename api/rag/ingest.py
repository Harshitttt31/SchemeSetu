"""Load official scheme PDFs from corpus/raw/ and extract clean text.

Public API (consumed by the chunk step):
    iter_corpus(corpus_dir) -> Iterator[PageRecord]
    iter_pdf_pages(pdf_path) -> Iterator[PageRecord]

Each PageRecord is one (source_doc, page_number, text) triple. Pages that
extract poorly are logged with a warning so the file can be switched to
pdfplumber — nothing is ever silently dropped.
"""
from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# corpus/raw/ sits two levels up from this file: api/rag/ingest.py -> api/corpus/raw
CORPUS_RAW = Path(__file__).resolve().parents[1] / "corpus" / "raw"

# A page with fewer than this many characters is treated as "no text".
MIN_PAGE_CHARS = 20
# If fewer than this fraction of a doc's pages yield text, flag it for pdfplumber.
MIN_NONEMPTY_PAGE_RATIO = 0.5


@dataclass
class PageRecord:
    """One extracted page of one document."""

    source_doc: str   # file name, e.g. "mudra_guidelines.pdf"
    page_number: int  # 1-based page index
    text: str         # cleaned text (may be "" if the page had no extractable text)


def clean_text(raw: str) -> str:
    """Normalise whitespace and strip common PDF extraction artifacts.

    Kept deliberately conservative: we remove noise but preserve paragraph
    breaks, because the chunker downstream relies on them.
    """
    if not raw:
        return ""
    text = raw.replace("\x0c", " ")     # form feed (page-break marker)
    text = text.replace(" ", " ")  # non-breaking space
    # Re-join words hyphenated across a line break: "regis-\ntration" -> "registration".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)          # collapse intra-line whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse blank-line runs
    text = "\n".join(line.strip() for line in text.splitlines())  # trim each line
    return text.strip()


def iter_pdf_pages(pdf_path: Path) -> Iterator[PageRecord]:
    """Yield cleaned PageRecords for one PDF; warn if it extracts poorly."""
    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    nonempty = 0

    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # one bad page shouldn't abort the document
            logger.warning("%s p%d: extract_text failed: %s", pdf_path.name, i, exc)
            raw = ""
        text = clean_text(raw)
        if len(text) >= MIN_PAGE_CHARS:
            nonempty += 1
        yield PageRecord(source_doc=pdf_path.name, page_number=i, text=text)

    if n_pages == 0:
        logger.warning("%s: PDF has 0 pages.", pdf_path.name)
    elif nonempty / n_pages < MIN_NONEMPTY_PAGE_RATIO:
        logger.warning(
            "%s: pypdf extracted text from only %d/%d pages -- this PDF likely "
            "needs pdfplumber. Switch this file over.",
            pdf_path.name, nonempty, n_pages,
        )


def iter_corpus(corpus_dir: Path = CORPUS_RAW) -> Iterator[PageRecord]:
    """Yield PageRecords for every PDF in the corpus directory (sorted by name)."""
    pdfs = sorted(corpus_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in %s", corpus_dir)
    for pdf_path in pdfs:
        yield from iter_pdf_pages(pdf_path)


def _summarise(corpus_dir: Path) -> None:
    """Acceptance-check helper: print doc count + a text sample per file."""
    pdfs = sorted(corpus_dir.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDF(s) in {corpus_dir}")
    if not pdfs:
        print("Place official scheme PDFs in corpus/raw/ and re-run.")
        return

    for pdf_path in pdfs:
        records = list(iter_pdf_pages(pdf_path))
        total_chars = sum(len(r.text) for r in records)
        first_nonempty = next((r for r in records if r.text), None)
        sample = ""
        if first_nonempty is not None:
            sample = re.sub(r"\s+", " ", first_nonempty.text)[:200]

        print(f"\n=== {pdf_path.name} ===")
        print(f"pages: {len(records)}  |  extracted chars: {total_chars}")
        if total_chars == 0:
            print("  WARNING: no text extracted from this PDF (see log above).")
        else:
            print(f"  first text on p{first_nonempty.page_number}: {sample!r}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Extract text from corpus PDFs.")
    parser.add_argument(
        "--corpus", type=Path, default=CORPUS_RAW,
        help="Directory of PDFs (default: api/corpus/raw).",
    )
    args = parser.parse_args()
    _summarise(args.corpus)
