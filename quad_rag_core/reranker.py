# qdrant_rag_core/reranker.py
from sentence_transformers import CrossEncoder
from typing import List, Tuple
import torch

class LocalReranker:
    _instance = None
    model: CrossEncoder
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = CrossEncoder(
                "BAAI/bge-reranker-v2-m3",
                max_length=512,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        return cls._instance

    def rerank(self, query: str, chunks: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Returns list of (chunk, score), sorted by relevance.
        """
        if not chunks:
            return []

        # Create pairs (query, chunk)
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        # Combine and sort
        results = list(zip(chunks, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]