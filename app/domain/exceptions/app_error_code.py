from enum import IntEnum


class AppErrorCode(IntEnum):
    """
    Error codes for application-specific exceptions.
    """

    ValidationFailed = 100
    CHAT_ERROR = 200
    KNOWLEDGE_BASE_ERROR = 201
    LLM_ERROR = 202
    VALIDATION_ERROR = 203
    SESSION_ERROR = 204
    InternalServerError = 500
    UnAuthorized = 401
