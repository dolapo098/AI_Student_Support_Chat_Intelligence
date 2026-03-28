from abc import ABC, abstractmethod
from typing import List

from app.contracts.dtos.chat_dtos import ChatMessageDto


class ILLMProvider(ABC):
    """
    Summary: Interface for Large Language Model providers.
    """

    @abstractmethod
    async def generate_response(
        self,
        message: str,
        context: str,
        history: List[ChatMessageDto],
        system_prompt: str
    ) -> str:
        """
        Summary: Generates a natural language response grounded in retrieved context.

        Args:
            message (str): The student's current message.
            context (str): Relevant Kent University document chunks from FAISS.
            history (List[ChatMessageDto]): Conversation history for this session.
            system_prompt (str): The Kent student support system instructions.

        Returns:
            str: The generated response text.
        """

        pass

    @abstractmethod
    async def detect_wellbeing(self, message: str) -> bool:
        """
        Summary: Detects whether a student message contains distress or wellbeing signals.

        Args:
            message (str): The student's message to analyse.

        Returns:
            bool: True if wellbeing concern detected, False otherwise.
        """

        pass
