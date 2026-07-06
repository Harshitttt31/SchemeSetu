"""Generate Gemini embeddings for chunks/queries with retry + exponential backoff.

Uses the official `google-genai` SDK. The model is a single config constant so it
can be swapped in one line if Google ships a newer embedding model.

Asymmetric embeddings improve retrieval: indexed chunks are embedded with
task_type=RETRIEVAL_DOCUMENT, live user queries with RETRIEVAL_QUERY.

Free-tier RPM is limited, so every call retries 429/5xx with exponential backoff.
"""
from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

logger = logging.getLogger(__name__)

# --- Config (verify against Google's docs; change here only) -----------------
EMBED_MODEL = "gemini-embedding-001"          # GA model, supports task_type
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"          # for indexed corpus chunks
TASK_QUERY = "RETRIEVAL_QUERY"                # for live user queries
DEFAULT_BATCH_SIZE = 32                       # keeps each request under token limits

# --- Backoff tuning ----------------------------------------------------------
_RETRY_CODES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 6
_BASE_DELAY = 2.0     # seconds
_MAX_DELAY = 60.0

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"  # api/.env


def get_client() -> genai.Client:
    """Build a Gemini client from GEMINI_API_KEY in api/.env (or the environment)."""
    load_dotenv(_ENV_PATH)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"GEMINI_API_KEY not set. Add it to {_ENV_PATH} (copy from .env.example)."
        )
    return genai.Client(api_key=api_key)


def _sleep_for(attempt: int) -> None:
    """Exponential backoff with jitter, capped at _MAX_DELAY."""
    delay = min(_MAX_DELAY, _BASE_DELAY * (2 ** attempt))
    delay += random.uniform(0, delay * 0.25)  # jitter to avoid thundering herd
    logger.warning("Rate-limited/transient error; backing off %.1fs (attempt %d).",
                   delay, attempt + 1)
    time.sleep(delay)


def _embed_batch(client: genai.Client, texts: list[str], task_type: str) -> list[list[float]]:
    """Embed one batch, retrying on 429/5xx with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in resp.embeddings]
        except errors.APIError as exc:
            last_exc = exc
            if getattr(exc, "code", None) in _RETRY_CODES and attempt < _MAX_ATTEMPTS - 1:
                _sleep_for(attempt)
                continue
            raise
    raise RuntimeError(f"Embedding failed after {_MAX_ATTEMPTS} attempts") from last_exc


def embed_texts(
    texts: list[str],
    *,
    task_type: str = TASK_DOCUMENT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    client: genai.Client | None = None,
) -> list[list[float]]:
    """Embed a list of texts (order preserved), batching to respect API limits."""
    if not texts:
        return []
    client = client or get_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info("Embedding %d-%d of %d", i + 1, i + len(batch), len(texts))
        out.extend(_embed_batch(client, batch, task_type))
    return out


def embed_query(text: str, *, client: genai.Client | None = None) -> list[float]:
    """Embed a single user query with the query task type."""
    return embed_texts([text], task_type=TASK_QUERY, client=client)[0]
