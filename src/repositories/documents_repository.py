import pandas as pd

from typing import List

from datasets import DatasetDict

from src.entities import Document

from .base import BaseRepository


class CsvDocumentsRepository(BaseRepository):
    def __init__(
        self,
        path: str,
        document_id_column: str = "document_id",
        document_content_column: str = "document_content",
    ) -> None:
        self.path = path
        self.document_id_column = document_id_column
        self.document_content_column = document_content_column

    def get_all(self) -> List[Document]:
        documents_df: pd.DataFrame = pd.read_csv(self.path).drop_duplicates(
            subset=[self.document_id_column, self.document_content_column]
        )
        return [
            Document(
                id=doc[self.document_id_column],
                content=doc[self.document_content_column],
                length=len(doc[self.document_content_column]),
            )
            for _, doc in documents_df.iterrows()
        ]


class DatasetDocumentsRepository(BaseRepository):
    def __init__(
        self,
        dataset: DatasetDict,
        document_id_column: str = "document_id",
        document_content_column: str = "document",
    ) -> None:
        self.dataset = dataset
        self.document_id_column = document_id_column
        self.document_content_column = document_content_column

    def get_all(self) -> List[Document]:
        seen_ids: set = set()
        documents: List[Document] = []
        for split in self.dataset.values():
            for row in split:
                doc_id = row[self.document_id_column]
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                content = row[self.document_content_column]
                documents.append(
                    Document(
                        id=doc_id,
                        content=content,
                        length=len(content),
                    )
                )
        return documents
