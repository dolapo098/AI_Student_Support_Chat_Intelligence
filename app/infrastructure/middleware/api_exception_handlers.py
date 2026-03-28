import logging
from fastapi import Request
from fastapi.responses import JSONResponse

from app.domain.exceptions.api_exception import ApiException

logger = logging.getLogger(__name__)


async def api_exception_handler(request: Request, exc: ApiException) -> JSONResponse:
    """
    Summary: Catches all ApiException subclasses and returns a consistent JSON error envelope.
    """

    logger.warning(
        "ApiException [%s] on %s: %s",
        exc.app_code.name,
        request.url.path,
        exc.message,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.app_code.value,
                "name": exc.app_code.name,
                "message": exc.message,
            }
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Summary: Catches any uncaught exception and returns a generic 500 response.
    """

    logger.error(
        "Unhandled exception on %s: %s", request.url.path, str(exc), exc_info=True
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "name": "InternalServerError",
                "message": "An unexpected error occurred. Please try again.",
            }
        },
    )
