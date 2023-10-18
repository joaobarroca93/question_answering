from typing import List, Optional

from src.entities import Document
from src.encoders import BaseEncoder
from src.clients import BaseDatabaseClient

from .base import BaseIndexer


class DatabaseIndexer(BaseIndexer):
    def __init__(self, client: BaseDatabaseClient, encoder: Optional[BaseEncoder] = None):
        self.client = client
        self.encoder = encoder

    def index(self, documents: List[Document]):
        if self.encoder:
            doc_contents = [doc.content for doc in documents]
            doc_vectors = self.encoder.batch_encode(texts=doc_contents)
            for i, doc in enumerate(documents):
                doc.vector = doc_vectors[i]
        self.client.add_documents(documents)
