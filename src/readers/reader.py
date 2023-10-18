from typing import List
from transformers import pipeline

from src.entities import Answer, RetrievalResult

from .base import BaseReader


class Reader(BaseReader):
    def __init__(self, model_filepath: str):
        self.qa_model = pipeline("question-answering", model=model_filepath)

    def extract(
        self,
        question: str,
        contexts: List[RetrievalResult],
        include_contexts: bool = False,
    ) -> List[Answer]:
        answers = []
        for context in contexts:
            answer = self.qa_model(question=question, context=context.document.content)  # type: ignore
            answers.append(
                Answer(
                    content=answer["answer"],
                    score=answer["score"],
                    context=context if include_contexts else None,
                )
            )
        return answers
