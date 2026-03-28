from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ChatMessageDto(BaseModel):
    """
    Summary: Represents a single message in the conversation history.
    """

    role: str

    content: str

    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """
    Summary: Request payload for the student chat endpoint.
    Conversation turns are stored server-side by ``session_id``; clients do not
    send message history (avoids conflicting sources of truth).
    """

    message: str = Field(..., min_length=1)

    session_id: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    """
    Summary: Response payload from the student chat endpoint.
    """

    answer: str

    session_id: str

    is_wellbeing: bool = False

    suggested_topics: List[str] = Field(default_factory=list)


class ClearSessionResponse(BaseModel):
    """
    Summary: Response payload after clearing a chat session.
    """

    message: str

    session_id: str


class HealthResponse(BaseModel):
    """
    Summary: Response payload for the health check endpoint.
    """

    status: str

    version: str

    knowledge_base_loaded: bool
