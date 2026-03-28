import asyncio
import logging
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.contracts.dtos.chat_dtos import ChatMessageDto
from app.contracts.providers.i_llm_provider import ILLMProvider
from app.domain.exceptions.chat_exception import LLMException

logger = logging.getLogger(__name__)

_RATE_LIMIT_RETRY_SECONDS = 15
_MAX_RETRIES = 2


class OpenAIProvider(ILLMProvider):
    """
    Summary: LangChain-based OpenAI provider for chat response generation.
    Supports gpt-4o-mini (default), gpt-4o, and other OpenAI chat models.
    Includes automatic retry on 429 rate-limit errors with backoff.
    """

    _WELLBEING_KEYWORDS = [
        "depressed", "depression", "anxious", "anxiety", "suicidal", "suicide",
        "self-harm", "self harm", "hurting myself", "can't cope", "cant cope",
        "overwhelmed", "mental health crisis", "breakdown", "hopeless",
        "don't want to be here", "dont want to be here", "stressed out",
        "panic attack", "eating disorder",
    ]

    def __init__(self, model_name: str, api_key: str):
        self._model_name = model_name
        self._api_key = api_key

        self._llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.3,
            max_tokens=1024,
        )

    async def generate_response(
        self,
        message: str,
        context: str,
        history: List[ChatMessageDto],
        system_prompt: str,
    ) -> str:

        messages = [SystemMessage(content=system_prompt)]

        for msg in history[-6:]:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))

        grounded_message = (
            f"Context from University of Kent knowledge base:\n"
            f"---\n{context}\n---\n\n"
            f"Student question: {message}"
        )

        messages.append(HumanMessage(content=grounded_message))

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._llm.ainvoke(messages)
                return response.content

            except Exception as exc:
                error_str = str(exc)

                if "429" in error_str and attempt < _MAX_RETRIES:
                    wait = _RATE_LIMIT_RETRY_SECONDS * (attempt + 1)
                    logger.warning(
                        "OpenAI rate limit hit (attempt %d/%d) — retrying in %ds",
                        attempt + 1, _MAX_RETRIES, wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                logger.error("OpenAI failed to generate response: %s", exc)
                raise LLMException(
                    f"LLM response generation failed: {str(exc)}", status_code=503
                )

    async def detect_wellbeing(self, message: str) -> bool:

        message_lower = message.lower()

        for keyword in self._WELLBEING_KEYWORDS:
            if keyword in message_lower:
                return True

        return False
