from typing import List

from src.entities import RetrievalResult, Answer
from src.retriever import BaseRetriever
from src.readers import BaseReader

from .base import BasePipeline


class ExtractiveQAPipeline(BasePipeline):
    def __init__(self, retriever: BaseRetriever, reader: BaseReader):
        self.retriever = retriever
        self.reader = reader

    def run(self, question: str, top_k: int = 3, include_contexts: bool = False) -> List[Answer]:
        contexts: List[RetrievalResult] = self.retriever.retrieve(query=question, k=top_k)
        return self.reader.extract(question=question, contexts=contexts, include_contexts=include_contexts)
