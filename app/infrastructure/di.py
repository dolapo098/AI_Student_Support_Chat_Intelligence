import logging
import os
from functools import lru_cache
from typing import Any, Optional

from app.contracts.providers.i_knowledge_provider import IKnowledgeProvider
from app.contracts.providers.i_llm_provider import ILLMProvider
from app.contracts.services.i_chat_service import IChatService
from app.infrastructure.factories.llm_provider_factory import LLMProviderFactory
from app.infrastructure.providers.faiss_knowledge_provider import FAISSKnowledgeProvider
from app.application.services.chat_service import ChatService
from app.infrastructure.security.pii_scrubber import PIIScrubber
from app.infrastructure.security.sanitizer_settings import (
    SanitizerSettings,
    load_sanitizer_settings,
)

logger = logging.getLogger(__name__)


def _load_spacy_for_settings(settings: SanitizerSettings) -> Optional[Any]:
    if not settings.ner_enabled:
        return None
    try:
        import spacy
    except ImportError:
        logger.warning(
            "ENABLE_SPACY_NER is set but spaCy is not installed; using regex-only scrubbing."
        )
        return None
    try:
        nlp = spacy.load(settings.spacy_model)
    except OSError:
        logger.warning(
            "SpaCy model %r not found; using regex-only scrubbing. "
            "Run: python -m spacy download %s",
            settings.spacy_model,
            settings.spacy_model,
        )
        return None
    logger.info("SpaCy NER loaded for PII scrubbing (%s)", settings.spacy_model)
    return nlp


@lru_cache(maxsize=1)
def _get_pii_scrubber() -> PIIScrubber:
    settings = load_sanitizer_settings()
    nlp = _load_spacy_for_settings(settings)
    return PIIScrubber(settings=settings, nlp=nlp)


@lru_cache(maxsize=1)
def _get_llm_provider() -> ILLMProvider:

    logger.info("Initialising LLM provider via factory.")

    return LLMProviderFactory.create()


@lru_cache(maxsize=1)
def _get_knowledge_provider() -> IKnowledgeProvider:

    api_key = os.getenv("GEMINI_API_KEY", "")
    index_path = os.getenv("FAISS_INDEX_PATH", "faiss_index")

    logger.info("Initialising FAISS knowledge provider at: %s", index_path)

    return FAISSKnowledgeProvider(api_key=api_key, index_path=index_path)


@lru_cache(maxsize=1)
def _get_chat_service() -> IChatService:

    llm = _get_llm_provider()
    knowledge = _get_knowledge_provider()

    scrubber = _get_pii_scrubber()

    logger.info("Initialising ChatService.")

    return ChatService(
        llm_provider=llm,
        knowledge_provider=knowledge,
        pii_scrubber=scrubber,
    )


def get_llm_provider() -> ILLMProvider:
    """FastAPI dependency — returns the singleton LLM provider."""
    return _get_llm_provider()


def get_knowledge_provider() -> IKnowledgeProvider:
    """FastAPI dependency — returns the singleton FAISS knowledge provider."""
    return _get_knowledge_provider()


def get_chat_service() -> IChatService:
    """FastAPI dependency — returns the singleton ChatService."""
    return _get_chat_service()


def get_pii_scrubber() -> PIIScrubber:
    """Singleton PII scrubber (settings + optional SpaCy loaded at first use)."""
    return _get_pii_scrubber()
