import joblib
from typing import List, Optional

from sklearn.pipeline import Pipeline

from src.processors.text import BaseTextProcessor

from .base import BaseEncoder


class TfIdfEncoder(BaseEncoder):
    def __init__(self, model_filepath: str, text_processor: Optional[BaseTextProcessor]):
        self.model = self._load_model(filepath=model_filepath)
        self.text_processor = text_processor

    @staticmethod
    def _load_model(filepath: str) -> Pipeline:
        return joblib.load(filename=filepath)

    def encode(self, text: str) -> List[float]:
        if self.text_processor:
            text = self.text_processor.process(text)
        return self.model.transform([text])[0].tolist()

    def batch_encode(self, texts: List[str]) -> List[List[float]]:
        if self.text_processor:
            texts = [self.text_processor.process(text) for text in texts]
        return self.model.transform(texts).tolist()
