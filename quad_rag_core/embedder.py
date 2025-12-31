# qdrant_rag_core/embedder.py
import torch
from sentence_transformers import SentenceTransformer
from typing import List

class LocalEmbedder:
    _instance = None
    model: SentenceTransformer
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v2-moe",
                trust_remote_code=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        return cls._instance

    def embed_document(self, text: str) -> List[float]:
        """Embed a document (for indexing)."""
        vector = self.model.encode(
            text,
            prompt_name="passage",
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return vector.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a search query (for inference)."""
        vector = self.model.encode(
            text,
            prompt_name="query",
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return vector.tolist()