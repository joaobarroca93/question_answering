import random

from typing import List

from src.clients import BaseDatabaseClient
from src.entities import RetrievalResult

from .base import BaseRetriever


class RandomRetriever(BaseRetriever):
    def __init__(self, client: BaseDatabaseClient, random_state: int = 42) -> None:
        self.client = client
        random.seed(random_state)

    def retrieve(self, query: str, k: int = 1) -> List[RetrievalResult]:
        documents = self.client.get_all_documents()
        results = random.choices(documents, k=k)
        return [RetrievalResult(document=doc, relevance=1.0) for doc in results]
