import logging
from typing import List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.contracts.dtos.chat_dtos import ChatMessageDto
from app.contracts.providers.i_llm_provider import ILLMProvider
from app.domain.exceptions.chat_exception import LLMException

logger = logging.getLogger(__name__)


class GroqProvider(ILLMProvider):
    """
    Summary: LangChain-based Groq provider for fast, free LLM response generation.
    Uses llama-3.3-70b-versatile by default - high quality, very fast via Groq's LPU.
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

        self._llm = ChatGroq(
            model=model_name,
            groq_api_key=api_key,
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

        try:
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

            response = await self._llm.ainvoke(messages)

            return response.content

        except Exception as exc:
            logger.error("Groq failed to generate response: %s", exc)
            raise LLMException(
                f"LLM response generation failed: {str(exc)}", status_code=503
            )

    async def detect_wellbeing(self, message: str) -> bool:

        message_lower = message.lower()

        for keyword in self._WELLBEING_KEYWORDS:
            if keyword in message_lower:
                return True

        return False
