import logging
from fastapi import Depends

from app.contracts.dtos.chat_dtos import (
    ChatRequest,
    ChatResponse,
    ClearSessionResponse,
    HealthResponse,
)
from app.contracts.providers.i_knowledge_provider import IKnowledgeProvider
from app.contracts.services.i_chat_service import IChatService
from app.infrastructure.di import get_chat_service, get_knowledge_provider

logger = logging.getLogger(__name__)


class ChatController:
    """
    Summary: Handles HTTP-level logic for the chat endpoints, delegating to the ChatService.
    """

    def __init__(
        self,
        chat_service: IChatService = Depends(get_chat_service),
        knowledge_provider: IKnowledgeProvider = Depends(get_knowledge_provider),
    ):
        self._chat_service = chat_service
        self._knowledge_provider = knowledge_provider

    async def handle_chat(self, request: ChatRequest) -> ChatResponse:

        logger.debug("Chat request received - session: %s", request.session_id)

        return await self._chat_service.chat(request)

    def handle_clear_session(self, session_id: str) -> ClearSessionResponse:

        logger.debug("Clear session request - session: %s", session_id)

        return self._chat_service.clear_session(session_id)

    def handle_health(self) -> HealthResponse:

        return HealthResponse(
            status="ok",
            version="1.0.0",
            knowledge_base_loaded=self._knowledge_provider.is_loaded(),
        )
