"""PIIScrubber: optional NER on raw text, then regex; regex-only helper for tests."""

import os

import pytest

from app.infrastructure.security.pii_scrubber import PIIScrubber
from app.infrastructure.security.sanitizer_settings import SanitizerSettings


def _settings(*, ner_enabled: bool = False) -> SanitizerSettings:
    return SanitizerSettings(
        sanitize_logs=True,
        ner_enabled=ner_enabled,
        ner_labels=frozenset({"PERSON"}),
        spacy_model="en_core_web_sm",
    )


def test_regex_redact_structured_pii():
    raw = "Reach me at x@kent.ac.uk in CT2 7NZ or 07700900111 id 12345678"
    scrub = PIIScrubber(_settings(), nlp=None)
    out = scrub.regex_only(raw)
    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_POSTCODE]" in out
    assert "[REDACTED_PHONE]" in out
    assert "[REDACTED_ID]" in out
    assert "kent.ac.uk" not in out.lower()


def test_for_llm_regex_without_ner():
    scrub = PIIScrubber(_settings(ner_enabled=False), nlp=None)
    out = scrub.for_llm("Only email test@example.com here")
    assert out == "Only email [REDACTED_EMAIL] here"


@pytest.mark.skipif(
    os.getenv("RUN_SPACY_SANITIZER_TESTS", "").lower() not in ("1", "true", "yes"),
    reason="Set RUN_SPACY_SANITIZER_TESTS=1 with model en_core_web_sm installed",
)
def test_for_llm_with_ner_person():
    pytest.importorskip("spacy")
    try:
        import spacy

        nlp = spacy.load(os.getenv("SPACY_MODEL", "en_core_web_sm"))
    except OSError as exc:
        pytest.skip(f"SpaCy English model not available: {exc}")

    scrub = PIIScrubber(_settings(ner_enabled=True), nlp=nlp)
    out = scrub.for_llm(
        "Can Alice Smith help with my extension? alice@kent.ac.uk"
    )
    assert "[REDACTED_EMAIL]" in out
    assert "alice@kent.ac.uk" not in out.lower()
    assert "[REDACTED_NAME]" in out
