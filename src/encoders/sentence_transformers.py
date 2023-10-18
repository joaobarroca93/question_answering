from typing import List

from sentence_transformers import SentenceTransformer

from .base import BaseEncoder


class SentenceTransformersEncoder(BaseEncoder):
    def __init__(self, model_filepath: str):
        self.model = self._load_model(filepath=model_filepath)

    @staticmethod
    def _load_model(filepath: str) -> SentenceTransformer:
        return SentenceTransformer(filepath)

    def encode(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

    def batch_encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
