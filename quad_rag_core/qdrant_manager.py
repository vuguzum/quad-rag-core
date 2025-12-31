# qdrant_rag_core/qdrant_client.py
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any
import os

class QdrantManager:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)

    def ensure_collection(self, collection_name: str, vector_size: int = 768):
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

    def upsert_points(self, collection_name: str, points: List[PointStruct]):
        self.client.upsert(collection_name=collection_name, points=points)

    def delete_by_path(self, collection_name: str, file_path: str):
        self.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
            must=[
                FieldCondition(
                    key="path",
                    match=MatchValue(value=file_path)
                )
            ]
        )
    )
        
    def delete_collection(self, collection_name: str):
            """
            Deletes entire collection from Qdrant
            """
            try:
                self.client.delete_collection(collection_name)
                print(f"[INFO] Collection '{collection_name}' successfully deleted.")
            except Exception as e:
                print(f"[ERROR] Failed to delete collection '{collection_name}': {e}")

    def search(self, collection_name: str, vector: List[float], limit: int = 10):
        """Search nearest vectors using query_points"""
        try:
            response = self.client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return response.points  # List of ScoredPoint
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return []