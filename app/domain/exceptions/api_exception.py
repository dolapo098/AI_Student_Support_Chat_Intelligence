from app.domain.exceptions.app_error_code import AppErrorCode


class ApiException(Exception):
    """
    Base exception class for all API-related errors in the domain.
    """

    def __init__(self, message: str, status_code: int, app_code: AppErrorCode):
        super().__init__(message)

        self.status_code = status_code
        self.app_code = app_code

    @property
    def message(self) -> str:

        return str(self)
