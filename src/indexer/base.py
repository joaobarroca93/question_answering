from typing import List
from abc import ABC, abstractmethod

from src.entities import Document
from src.clients import BaseDatabaseClient


class BaseIndexer(ABC):
    def __init__(self, client: BaseDatabaseClient):
        self.client = client

    @abstractmethod
    def index(self, documents: List[Document]):
        raise NotImplementedError
