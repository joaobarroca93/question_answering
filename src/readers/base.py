from typing import List
from abc import ABC, abstractmethod

from src.entities import Answer, RetrievalResult


class BaseReader(ABC):
    @abstractmethod
    def extract(
        self,
        question: str,
        contexts: List[RetrievalResult],
        include_contexts: bool = False,
    ) -> List[Answer]:
        raise NotImplementedError

    def batch_extract(
        self,
        questions: List[str],
        batch_contexts: List[List[RetrievalResult]],
        include_contexts: bool = False,
    ) -> List[List[Answer]]:
        return [
            self.extract(question=question, contexts=contexts, include_contexts=include_contexts)
            for question, contexts in zip(questions, batch_contexts)
        ]
