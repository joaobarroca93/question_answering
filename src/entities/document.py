from typing import Any, Dict, Sequence, Mapping, Optional, Union
from dataclasses import dataclass

METADATA = Mapping[str, Union[str, int, float, bool]]
VECTOR = Union[Sequence[float], Sequence[int]]


@dataclass
class Document:
    id: str
    content: str
    length: int
    vector: Optional[VECTOR] = None
    metadata: Optional[METADATA] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "length": self.length,
            "metadata": self.metadata,
        }
