from abc import ABC, abstractmethod

from app.contracts.dtos.chat_dtos import ChatRequest, ChatResponse, ClearSessionResponse


class IChatService(ABC):
    """
    Summary: Interface for the main student support chat service.
    """

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Summary: Processes a student message and returns a grounded, contextual response.

        Args:
            request (ChatRequest): The incoming chat request (message and session ID).
                                   Prior turns are read from the server-side session store.

        Returns:
            ChatResponse: The AI response, session ID, wellbeing flag, and suggested topics.
        """

        pass

    @abstractmethod
    def clear_session(self, session_id: str) -> ClearSessionResponse:
        """
        Summary: Clears the conversation history for a given session.

        Args:
            session_id (str): The session to clear.

        Returns:
            ClearSessionResponse: Confirmation message and session ID.
        """

        pass
