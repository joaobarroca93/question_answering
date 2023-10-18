import nltk
import unicodedata
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from .base import BaseTextProcessor

nltk.download("punkt")
nltk.download("stopwords")


class TextProcessor(BaseTextProcessor):
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r"\w+")
        self.ps = PorterStemmer()
        self.stopwords = stopwords.words("english")

    def process(self, text: str) -> str:
        # lowercasing
        text = text.lower()
        # accents normalization
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        # stemming
        tokens = [self.ps.stem(token) for token in self.tokenizer.tokenize(text)]
        # stopwords removal
        tokens = [token for token in tokens if token not in self.stopwords]
        return " ".join(tokens)
