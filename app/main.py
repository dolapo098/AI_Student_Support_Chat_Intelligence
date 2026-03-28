import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.domain.exceptions.api_exception import ApiException
from app.infrastructure.di import get_chat_service
from app.infrastructure.middleware.api_exception_handlers import (
    api_exception_handler,
    unhandled_exception_handler,
)
from app.infrastructure.middleware.security_headers_middleware import (
    SecurityHeadersMiddleware,
)
from app.routes.chat_routes import router as chat_router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

_INDEX_DIR = Path(os.getenv("FAISS_INDEX_PATH", "faiss_index"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-warms singleton services on startup.
    Build the FAISS index separately: python scripts/build_knowledge_base.py
    """

    index_file = _INDEX_DIR / "index.faiss"
    chunks_file = _INDEX_DIR / "chunks.pkl"

    if not index_file.exists() or not chunks_file.exists():
        logger.warning(
            "No FAISS knowledge base at %s. Run: python scripts/build_knowledge_base.py",
            _INDEX_DIR,
        )

    logger.info("Pre-warming AI services...")
    get_chat_service()
    logger.info("All services ready — Kay is online.")

    yield

    logger.info("Kay is shutting down.")


app = FastAPI(
    title="Kay — University of Kent Student Support AI",
    description=(
        "AI-powered student support chatbot for the University of Kent. "
        "Answers queries on admissions, assessments, deadlines, wellbeing, and general enquiries."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

_default_origins = (
    "http://localhost:3000,http://localhost:5173,http://localhost:5174,"
    "http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:5174"
)
_allowed_origins = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)

app.add_exception_handler(ApiException, api_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(chat_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Kay — Kent Student Support AI",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }
