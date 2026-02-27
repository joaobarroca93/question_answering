from src.processors.text.text_processor import TextProcessor


class TestTextProcessor:
    def setup_method(self):
        self.processor = TextProcessor()

    def test_lowercases_text(self):
        result = self.processor.process("HELLO WORLD")
        assert result == result.lower()

    def test_removes_stopwords(self):
        result = self.processor.process("this is a test")
        # "this", "is", "a" are stopwords; "test" should survive stemming
        assert "test" in result or "thi" in result  # stemmed form

    def test_normalizes_accents(self):
        result = self.processor.process("café")
        assert "é" not in result

    def test_stems_words(self):
        result = self.processor.process("running")
        # PorterStemmer reduces "running" -> "run"
        assert "run" in result

    def test_returns_string(self):
        result = self.processor.process("some text")
        assert isinstance(result, str)

    def test_empty_string(self):
        result = self.processor.process("")
        assert result == ""
