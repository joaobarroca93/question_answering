from typing import List
from abc import ABC, abstractmethod

from src.entities import RetrievalResult
from src.clients.base import BaseDatabaseClient


class BaseRetriever(ABC):
    def __init__(self, client: BaseDatabaseClient) -> None:
        self.client = client

    @abstractmethod
    def retrieve(self, query: str, k: int = 1) -> List[RetrievalResult]:
        raise NotImplementedError

    def batch_retrieve(self, queries: List[str], k: int = 1) -> List[List[RetrievalResult]]:
        return [self.retrieve(query) for query in queries]
