from typing import List
from abc import ABC, abstractmethod


class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, text: str) -> List[float]:
        raise NotImplementedError

    def batch_encode(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(text) for text in texts]
