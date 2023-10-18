from abc import ABC, abstractmethod


class BaseTextProcessor(ABC):
    @abstractmethod
    def process(self, text: str) -> str:
        raise NotImplementedError
