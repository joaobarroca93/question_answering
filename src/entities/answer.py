from typing import Any, Dict, Optional
from dataclasses import dataclass

from .retrieval_result import RetrievalResult


@dataclass
class Answer:
    content: str
    score: float
    context: Optional[RetrievalResult] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.content,
            "score": self.score,
            "context": self.context.as_dict() if self.context else None,
        }
