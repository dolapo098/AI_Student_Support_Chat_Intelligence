# Kay — Kent Student Support API (FastAPI + FAISS RAG).
# Railway sets PORT at runtime; default 8001 for local: docker run -p 8001:8001 ...

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install CPU-only PyTorch first so sentence-transformers does not pull CUDA wheels (~GB).
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY app ./app

# Repo tracks faiss_index/.gitkeep so COPY always succeeds; add index.faiss + chunks.pkl before production deploy.
COPY faiss_index ./faiss_index

EXPOSE 8001

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
