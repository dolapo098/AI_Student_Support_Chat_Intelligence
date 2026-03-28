"""
Type-safe, immutable sanitizer configuration.

``SanitizerSettings`` is a frozen dataclass; ``load_sanitizer_settings`` is
``lru_cache``(maxsize=1) so values are read from the environment once per process
and the PII scrubber always sees a stable config without per-request ``getenv``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import FrozenSet

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


_DEFAULT_NER_LABELS: FrozenSet[str] = frozenset({"PERSON"})


def _parse_ner_labels(raw: str) -> FrozenSet[str]:
    try:
        labels = frozenset(p.strip().upper() for p in raw.split(",") if p.strip())
    except (TypeError, AttributeError):
        logger.warning("NER_REDACT_LABELS could not be parsed; using PERSON")
        return _DEFAULT_NER_LABELS
    if not labels:
        logger.warning("NER_REDACT_LABELS is empty after parse; using PERSON")
        return _DEFAULT_NER_LABELS
    return labels


@dataclass(frozen=True)
class SanitizerSettings:
    """Read once at process startup; passed into PIIScrubber."""

    sanitize_logs: bool
    ner_enabled: bool
    ner_labels: FrozenSet[str]
    spacy_model: str


@lru_cache(maxsize=1)
def load_sanitizer_settings() -> SanitizerSettings:
    labels_raw = (os.getenv("NER_REDACT_LABELS") or "PERSON").strip()
    ner_labels = _parse_ner_labels(labels_raw) if labels_raw else _DEFAULT_NER_LABELS
    model = (os.getenv("SPACY_MODEL") or "en_core_web_sm").strip() or "en_core_web_sm"
    return SanitizerSettings(
        sanitize_logs=_env_bool("SANITIZE_LOGS", True),
        ner_enabled=_env_bool("ENABLE_SPACY_NER", False),
        ner_labels=ner_labels,
        spacy_model=model,
    )
