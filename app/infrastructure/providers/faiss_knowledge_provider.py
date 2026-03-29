import logging
import os
import pickle
import sys
from pathlib import Path
from typing import List

# Set before importing faiss / numpy / torch stacks (Windows OpenMP errno 22 mitigations).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.contracts.dtos.retrieval_dtos import RetrievedChunk
from app.contracts.providers.i_knowledge_provider import IKnowledgeProvider
from app.domain.exceptions.chat_exception import KnowledgeBaseException

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class FAISSKnowledgeProvider(IKnowledgeProvider):
    """
    Summary: Loads and searches a FAISS vector index built from Kent University documents.
    Uses a local sentence-transformers model for embeddings - no API key required.
    """

    def __init__(self, api_key: str, index_path: str = "faiss_index"):
        self._index_path = Path(index_path)
        self._index: faiss.Index | None = None
        self._chunks: List[str] = []

        logger.info("Loading local embedding model: %s", _EMBEDDING_MODEL)
        self._embedder = SentenceTransformer(_EMBEDDING_MODEL)

        self._load_index()

    def _load_index(self) -> None:

        index_file = self._index_path / "index.faiss"
        chunks_file = self._index_path / "chunks.pkl"

        if not index_file.exists() or not chunks_file.exists():
            logger.warning(
                "FAISS index not found at %s - will build on first request.", self._index_path
            )
            return

        try:
            # Load index bytes into memory then deserialize - avoids mmap on the file.
            # On Windows + OneDrive/cloud folders, mmap can raise OSError [Errno 22] Invalid argument.
            raw = index_file.read_bytes()
            data = np.frombuffer(raw, dtype=np.uint8).copy()

            try:
                self._index = faiss.deserialize_index(data)
            except Exception:
                logger.warning("deserialize_index failed; falling back to read_index.")
                self._index = faiss.read_index(str(index_file))

            with open(chunks_file, "rb") as f:
                self._chunks = pickle.load(f)

            logger.info("FAISS index loaded: %d chunks ready.", len(self._chunks))

        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            raise KnowledgeBaseException(f"Failed to load knowledge base: {str(exc)}")

    def _numpy_flat_l2_topk(self, q: np.ndarray, k: int) -> np.ndarray:
        """
        Brute-force top-k by L2 distance using vectors reconstructed from the index.
        Used when faiss.search fails on some Windows environments (e.g. Errno 22).
        """
        idx = self._index
        if idx is None or not hasattr(idx, "reconstruct"):
            raise KnowledgeBaseException(
                "Knowledge base search fallback is not supported for this index type."
            )

        ntotal = int(idx.ntotal)
        d = int(idx.d)
        k = min(k, ntotal)
        if k <= 0:
            return np.array([], dtype=np.int64)

        qv = np.ascontiguousarray(q.reshape(-1), dtype=np.float32)
        matrix = np.empty((ntotal, d), dtype=np.float32)
        for i in range(ntotal):
            matrix[i] = idx.reconstruct(i)

        dists = np.sum((matrix - qv) ** 2, axis=1)
        if k >= ntotal:
            order = np.argsort(dists)
        else:
            partial = np.argpartition(dists, k - 1)[:k]
            order = partial[np.argsort(dists[partial])]
        return order.astype(np.int64)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:

        if self._index is None or not self._chunks:
            logger.warning("Knowledge base is not loaded - returning empty context.")
            return []

        if not query.strip():
            return []

        try:
            try:
                query_embedding = self._embedder.encode(
                    query,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            except OSError as enc_err:
                if getattr(enc_err, "errno", None) == 22:
                    logger.warning(
                        "encode() raised errno 22 - retrying with batch_size=1."
                    )
                    query_embedding = self._embedder.encode(
                        query,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=1,
                    )
                else:
                    raise

            # FAISS on Windows expects C-contiguous float32 (avoids subtle search errors).
            q = np.ascontiguousarray(
                np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
            )

            ntotal = int(self._index.ntotal)
            if ntotal <= 0:
                return []

            expected_d = int(self._index.d)
            if q.shape[1] != expected_d:
                raise KnowledgeBaseException(
                    f"Embedding dimension {q.shape[1]} does not match index ({expected_d})."
                )

            k = min(top_k, ntotal, len(self._chunks))
            if k <= 0:
                return []

            # faiss.search often raises OSError [Errno 22] on Windows (pip faiss-cpu + OpenMP).
            # For modest index sizes, pure NumPy is fast and reliable.
            use_numpy = sys.platform == "win32" and ntotal <= 20_000
            if use_numpy:
                idx_row = self._numpy_flat_l2_topk(q, k)
            else:
                try:
                    _, indices = self._index.search(q, k)
                    idx_row = indices[0]
                except OSError as exc:
                    logger.warning(
                        "FAISS index.search OSError (%s); using NumPy L2 fallback.", exc
                    )
                    idx_row = self._numpy_flat_l2_topk(q, k)
                except Exception as exc:
                    logger.warning(
                        "FAISS index.search failed (%s); using NumPy L2 fallback.", exc
                    )
                    idx_row = self._numpy_flat_l2_topk(q, k)

            results: List[RetrievedChunk] = []
            for idx in idx_row:
                i = int(idx)
                if 0 <= i < len(self._chunks):
                    results.append(RetrievedChunk(chunk_index=i, text=self._chunks[i]))

            return results

        except KnowledgeBaseException:
            raise
        except Exception as exc:
            logger.exception("FAISS search failed: %s", exc)
            raise KnowledgeBaseException(f"Knowledge base search failed: {str(exc)}")

    def is_loaded(self) -> bool:

        return self._index is not None and len(self._chunks) > 0
