from app.domain.exceptions.app_error_code import AppErrorCode
from app.domain.exceptions.api_exception import ApiException


class ChatException(ApiException):
    """
    Base class for all chat-related exceptions.
    """

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code, AppErrorCode.CHAT_ERROR)


class LLMException(ChatException):
    """
    Raised when the LLM provider fails to generate a response.
    """

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code)

        self.app_code = AppErrorCode.LLM_ERROR


class KnowledgeBaseException(ChatException):
    """
    Raised when the FAISS knowledge base fails to load or search.
    """

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code)

        self.app_code = AppErrorCode.KNOWLEDGE_BASE_ERROR


class InvalidChatRequestException(ChatException):
    """
    Raised when a chat request is malformed or invalid.
    """

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code)

        self.app_code = AppErrorCode.VALIDATION_ERROR


class SessionException(ChatException):
    """
    Raised when a session operation fails.
    """

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code)

        self.app_code = AppErrorCode.SESSION_ERROR
