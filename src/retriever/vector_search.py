from typing import List

from src.entities import RetrievalResult
from src.encoders.base import BaseEncoder
from src.clients.base import BaseVectorDatabaseClient

from .base import BaseRetriever


class VectorSearchRetriever(BaseRetriever):
    def __init__(self, client: BaseVectorDatabaseClient, encoder: BaseEncoder) -> None:
        self.client: BaseVectorDatabaseClient = client
        self.encoder: BaseEncoder = encoder

    def retrieve(self, query: str, k: int = 1) -> List[RetrievalResult]:
        query_vector: List[float] = self.encoder.encode(text=query)
        results = self.client.query(query_vectors=[query_vector], k=k)
        docs, distances = results[0]
        return [
            RetrievalResult(document=document, relevance=1 - distance) for document, distance in zip(docs, distances)
        ]

    def batch_retrieve(self, queries: List[str], k: int = 1) -> List[List[RetrievalResult]]:
        query_vectors: List[List[float]] = self.encoder.batch_encode(texts=queries)
        results = self.client.query(query_vectors=query_vectors, k=k)
        return [
            [RetrievalResult(document=document, relevance=1 - distance) for document, distance in zip(docs, distances)]
            for docs, distances in results
        ]
