from typing import Dict, List, cast
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
            raw = cast(Dict[str, object], self.qa_model(question=question, context=context.document.content))
            answers.append(
                Answer(
                    content=cast(str, raw["answer"]),
                    score=cast(float, raw["score"]),
                    context=context if include_contexts else None,
                )
            )
        return answers
