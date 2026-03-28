# Kay — Kent Student Support API (FastAPI + FAISS RAG).
# Knowledge base is built during image build (scrapes Kent URLs + embeddings) so GitHub CI and Railway
# both ship a working RAG index without committing index.faiss / chunks.pkl.
# Railway sets PORT at runtime; local: docker run -p 8001:8001 ...

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

COPY scripts ./scripts
RUN python scripts/build_knowledge_base.py && \
    test -f faiss_index/index.faiss && \
    test -f faiss_index/chunks.pkl

COPY app ./app

EXPOSE 8001

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
