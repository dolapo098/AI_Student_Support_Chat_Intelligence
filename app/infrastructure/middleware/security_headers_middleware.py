import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


def _env_flag(name: str) -> bool:
    v = os.getenv(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds baseline HTTP security headers when ENABLE_SECURITY_HEADERS is set.

    HSTS is only attached when HSTS_MAX_AGE is a positive integer — enable this
    only when the app is actually served over HTTPS (e.g. behind TLS-terminated
    ingress) so browsers do not cache HSTS for plain HTTP dev URLs.
    """

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        if not _env_flag("ENABLE_SECURITY_HEADERS"):
            return response

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        hsts = os.getenv("HSTS_MAX_AGE", "").strip()
        if hsts.isdigit() and int(hsts) > 0:
            response.headers["Strict-Transport-Security"] = (
                f"max-age={int(hsts)}; includeSubDomains"
            )

        return response
