from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

from src.entities import Document


class BaseDatabaseClient(ABC):
    @abstractmethod
    def add_document(self, document: Document) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove_document(self, document_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        raise NotImplementedError

    @abstractmethod
    def get_all_documents(self) -> List[Document]:
        raise NotImplementedError

    def add_documents(self, documents: List[Document]) -> None:
        for document in documents:
            self.add_document(document=document)

    def remove_documents(self, document_ids: List[str]) -> None:
        for document_id in document_ids:
            self.remove_document(document_id=document_id)


class BaseVectorDatabaseClient(BaseDatabaseClient):
    @abstractmethod
    def query(self, query_vectors: List[List[float]], k: int = 2) -> List[Tuple[List[Document], List[float]]]:
        raise NotImplementedError
