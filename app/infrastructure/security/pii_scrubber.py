from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from app.infrastructure.security.sanitizer_settings import SanitizerSettings

_EMAIL = re.compile(
    r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
)
_UK_PHONE = re.compile(
    r"(?i)\b(?:\+44\s?7\d{2}|\(?07\d{2}\)?)[\s\-]?\d{3}[\s\-]?\d{4}\b"
    r"|\b(?:\+44\s?|0)(?:\d[\s\-]?){9,11}\d\b",
)
_UK_POSTCODE = re.compile(
    r"(?i)\b[A-Z]{1,2}\d[A-Z0-9]?\s*\d[A-Z]{2}\b",
)
_LONG_ID = re.compile(r"\b(?!(?:19|20)\d{2}\b)\d{7,10}\b")
_LABELLED_ID = re.compile(
    r"(?i)\b(?:student|stu|banner|sis|guid|urn|reference|ref)\s*[#:=]+\s*"
    r"[A-Za-z0-9][A-Za-z0-9_-]{2,}\b"
    r"|\bid\s*[#:=]+\s*[A-Za-z0-9][A-Za-z0-9_-]{2,}\b",
)

_NER_PLACEHOLDERS = {
    "PERSON": "[REDACTED_NAME]",
    "ORG": "[REDACTED_ORG]",
    "GPE": "[REDACTED_LOCATION]",
    "LOC": "[REDACTED_LOCATION]",
}

Span = Tuple[int, int, str]


class PIIScrubber:
    """
    Optional SpaCy NER on the **original** text, then regex for deterministic PII.
    NER never sees ``[REDACTED_*]`` placeholders (which can otherwise be misparsed
    as ORG). Holds no process-global state; ``nlp`` is injected from DI.
    """

    def __init__(
        self,
        settings: SanitizerSettings,
        nlp: Optional[Any] = None,
    ):
        self._settings = settings
        if not settings.ner_enabled:
            self._nlp = None
        else:
            self._nlp = nlp

    def for_llm(self, text: str) -> str:
        if not text:
            return text
        t = text
        if self._nlp is not None:
            t = self._ner_redact(t)
        return self._regex_redact(t)

    def for_logs(self, text: str) -> str:
        if not self._settings.sanitize_logs:
            return text
        return self.for_llm(text)

    @staticmethod
    def _regex_redact(text: str) -> str:
        t = _EMAIL.sub("[REDACTED_EMAIL]", text)
        t = _UK_PHONE.sub("[REDACTED_PHONE]", t)
        t = _UK_POSTCODE.sub("[REDACTED_POSTCODE]", t)
        t = _LABELLED_ID.sub("[REDACTED_ID]", t)
        t = _LONG_ID.sub("[REDACTED_ID]", t)
        return t

    def _ner_redact(self, text: str) -> str:
        """Run on raw student text only (before regex) so placeholders do not confuse NER."""
        doc = self._nlp(text)
        spans: List[Span] = [
            (e.start_char, e.end_char, e.label_)
            for e in doc.ents
            if e.label_ in self._settings.ner_labels
        ]
        if not spans:
            return text
        return _apply_spans(text, spans, _NER_PLACEHOLDERS)

    @staticmethod
    def regex_only(text: str) -> str:
        """Exposed for tests and callers that need the deterministic stage only."""
        return PIIScrubber._regex_redact(text)


def _apply_spans(
    text: str,
    spans: List[Span],
    placeholders: dict,
) -> str:
    """Single forward pass: build output from slices (avoids repeated full-string copies)."""
    spans = sorted(spans, key=lambda s: s[0])
    parts: List[str] = []
    cursor = 0
    for start, end, label in spans:
        if end <= cursor:
            continue
        if start < cursor:
            start = cursor
        parts.append(text[cursor:start])
        parts.append(placeholders.get(label, f"[REDACTED_{label}]"))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)
