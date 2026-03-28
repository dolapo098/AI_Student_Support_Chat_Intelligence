from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """One FAISS hit: index into chunks.pkl and the text passed to the LLM as context."""

    chunk_index: int
    text: str
