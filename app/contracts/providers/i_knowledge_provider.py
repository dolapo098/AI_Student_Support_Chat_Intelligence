from abc import ABC, abstractmethod
from typing import List

from app.contracts.dtos.retrieval_dtos import RetrievedChunk


class IKnowledgeProvider(ABC):
    """
    Summary: Interface for the FAISS knowledge base provider.
    """

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Summary: Performs semantic similarity search against the Kent knowledge base.

        Args:
            query (str): The student's query to search against.
            top_k (int): Number of most relevant document chunks to retrieve.

        Returns:
            Retrieved chunks with stable ``chunk_index`` into the index (for audit / RAG debugging).
        """

        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Summary: Indicates whether the FAISS index has been loaded and is ready.

        Returns:
            bool: True if the knowledge base is ready, False otherwise.
        """

        pass
