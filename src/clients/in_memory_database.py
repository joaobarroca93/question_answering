from typing import Dict, List, Optional

from src.entities import Document

from .base import BaseDatabaseClient


class InMemoryDatabaseClient(BaseDatabaseClient):
    def __init__(self, documents: Optional[List[Document]] = None):
        self.documents: Dict[str, Document] = {}

        if documents:
            self.add_documents(documents=documents)

    def add_document(self, document: Document) -> None:
        if document.id in self.documents.keys():
            raise ValueError("There is already a document with id `{document.id}`.")
        self.documents[document.id] = document

    def remove_document(self, document_id: str) -> None:
        if document_id not in self.documents.keys():
            raise ValueError("There is not any document with the provided id.")
        self.documents.pop(document_id)

    def get_document(self, document_id: str) -> Optional[Document]:
        return self.documents.get(document_id, None)

    def get_all_documents(self) -> List[Document]:
        return list(self.documents.values())
