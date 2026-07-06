"""Pydantic models for the structured, citation-bearing answer payload.

This is the strict contract every generated answer must satisfy. The generation
step (generate.py) asks Gemini for JSON in this shape; the validation step
(validate.py) re-parses the model's output through THESE models before we trust
it. The model's JSON is a claim -- this schema is where we make it prove itself.

STRUCTURED OUTPUT SHAPE:
    { "answer": str,
      "citations": [ {"chunk_id": str, "source": str} ],
      "confidence": "high" | "medium" | "low" }

Strictness enforced here (beyond field types):
  * No extra fields anywhere (extra="forbid").
  * confidence is a closed enum -- any other string is rejected.
  * citations may be empty ONLY for the canonical not-found response, and the
    not-found response may NOT carry citations. Empty-citations <-> not-found is
    a strict bijection, so an answer can never be simultaneously grounded and
    source-less, nor claim sources while admitting it found nothing.
  * a not-found response is always low confidence (we found nothing to be sure of).

Produce not-found responses via AnswerResponse.not_found() so the whole system
shares one exact sentinel string.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Canonical not-found answer. Kept as one constant so generate.py and validate.py
# never drift on the wording -- the empty-citations rule matches this exactly.
NOT_FOUND_ANSWER = "The answer is not available in the provided sources."


class Confidence(str, Enum):
    """Closed set of confidence levels. Any other value is a validation error."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Citation(BaseModel):
    """One source pointer: which retrieved chunk backs a claim, and its document."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1, description="ID of a chunk actually retrieved.")
    source: str = Field(min_length=1, description="Human-readable source document.")


class AnswerResponse(BaseModel):
    """The full grounded answer returned to the caller/UI."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence

    @property
    def is_not_found(self) -> bool:
        """True when this is the canonical not-found response."""
        return self.answer.strip() == NOT_FOUND_ANSWER

    @model_validator(mode="after")
    def _check_citation_grounding(self) -> "AnswerResponse":
        """Enforce the empty-citations <-> not-found bijection (+ low confidence)."""
        not_found = self.is_not_found

        if not self.citations and not not_found:
            raise ValueError(
                "citations may be empty ONLY for the not-found response; a grounded "
                "answer must cite at least one chunk."
            )
        if self.citations and not_found:
            raise ValueError(
                "the not-found response must not carry citations (it found nothing)."
            )
        if not_found and self.confidence is not Confidence.LOW:
            raise ValueError("the not-found response must be low confidence.")
        return self

    @classmethod
    def not_found(cls) -> "AnswerResponse":
        """The one true not-found response: no citations, low confidence."""
        return cls(answer=NOT_FOUND_ANSWER, citations=[], confidence=Confidence.LOW)
