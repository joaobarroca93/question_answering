from typing import Any, Dict
from dataclasses import dataclass

from .document import Document


@dataclass
class RetrievalResult:
    document: Document
    relevance: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document.as_dict(),
            "relevance": self.relevance,
        }
