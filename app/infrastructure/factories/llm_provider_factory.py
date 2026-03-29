import logging
import os

from app.contracts.providers.i_llm_provider import ILLMProvider
from app.infrastructure.providers.gemini_provider import GeminiProvider
from app.infrastructure.providers.groq_provider import GroqProvider
from app.infrastructure.providers.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS = {"gemini", "openai", "groq"}


class LLMProviderFactory:
    """
    Summary: Factory that creates the appropriate LLM provider based on environment config.
    Supports:
      - openai → OpenAI GPT (gpt-4o-mini recommended)
      - gemini → Google Gemini (gemini-2.5-flash recommended)
      - groq   → Groq LPU inference (llama-3.3-70b-versatile)
    Switch provider via LLM_PROVIDER env var - no code changes needed.
    """

    @staticmethod
    def create() -> ILLMProvider:

        provider_name = os.getenv("LLM_PROVIDER", "openai").lower()
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

        if provider_name not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider: '{provider_name}'. "
                f"Supported: {_SUPPORTED_PROVIDERS}"
            )

        logger.info(
            "Creating LLM provider: %s - model: %s", provider_name, model_name
        )

        if provider_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Please add it to your .env file."
                )
            return OpenAIProvider(model_name=model_name, api_key=api_key)

        if provider_name == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Please add it to your .env file."
                )
            return GeminiProvider(model_name=model_name, api_key=api_key)

        if provider_name == "groq":
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY is not set. Please add it to your .env file."
                )
            return GroqProvider(model_name=model_name, api_key=api_key)
