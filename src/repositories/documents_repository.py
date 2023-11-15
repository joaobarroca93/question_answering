import itertools
import pandas as pd

from typing import List

from datasets import DatasetDict

from src.entities.document import Document

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
        unique_doc_ids = self.dataset.unique(self.document_id_column).values()
        unique_doc_contents = self.dataset.unique(self.document_content_column).values()
        doc_ids = list(itertools.chain.from_iterable(unique_doc_ids))
        doc_contents = list(itertools.chain.from_iterable(unique_doc_contents))
        return [
            Document(
                id=doc_id,
                content=doc_content,
                length=len(doc_content),
            )
            for doc_id, doc_content in zip(doc_ids, doc_contents)
        ]
