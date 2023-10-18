import numpy as np

from typing import List

from .base import BaseEncoder


class RandomEncoder(BaseEncoder):
    def __init__(self, embedding_dim: int, random_state: int = 42):
        self.embedding_dim = embedding_dim
        np.random.seed(random_state)

    def encode(self, text: str) -> List[float]:
        return np.random.randn(self.embedding_dim).tolist()

    def batch_encode(self, texts: List[str]) -> List[List[float]]:
        return np.random.randn(len(texts), self.embedding_dim).tolist()
