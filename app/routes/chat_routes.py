from fastapi import APIRouter, Depends

from app.contracts.dtos.chat_dtos import (
    ChatRequest,
    ChatResponse,
    ClearSessionResponse,
    HealthResponse,
)
from app.contracts.providers.i_knowledge_provider import IKnowledgeProvider
from app.contracts.services.i_chat_service import IChatService
from app.controllers.chat_controller import ChatController
from app.infrastructure.di import get_chat_service, get_knowledge_provider

router = APIRouter(prefix="/chat", tags=["Chat"])


def get_controller(
    chat_service: IChatService = Depends(get_chat_service),
    knowledge_provider: IKnowledgeProvider = Depends(get_knowledge_provider),
) -> ChatController:
    return ChatController(
        chat_service=chat_service,
        knowledge_provider=knowledge_provider,
    )


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    controller: ChatController = Depends(get_controller),
) -> ChatResponse:
    """
    Summary: Accepts a student message and returns a grounded AI response from Kay.
    """

    return await controller.handle_chat(request)


@router.delete("/session/{session_id}", response_model=ClearSessionResponse)
def clear_session(
    session_id: str,
    controller: ChatController = Depends(get_controller),
) -> ClearSessionResponse:
    """
    Summary: Clears the conversation memory for the given session.
    """

    return controller.handle_clear_session(session_id)


@router.get("/health", response_model=HealthResponse)
def health_check(
    controller: ChatController = Depends(get_controller),
) -> HealthResponse:
    """
    Summary: Returns service health and whether the FAISS knowledge base is loaded.
    """

    return controller.handle_health()
