from unittest.mock import MagicMock

from src.pipelines.extractive_qa import ExtractiveQAPipeline
from src.entities.document import Document
from src.entities.retrieval_result import RetrievalResult
from src.entities.answer import Answer


def make_retrieval_result(doc_id: str = "d1", content: str = "Paris is the capital of France.") -> RetrievalResult:
    doc = Document(id=doc_id, content=content, length=len(content))
    return RetrievalResult(document=doc, relevance=0.9)


class TestExtractiveQAPipeline:
    def test_run_calls_retriever_and_reader(self):
        retriever = MagicMock()
        reader = MagicMock()
        contexts = [make_retrieval_result()]
        expected_answers = [Answer(content="Paris", score=0.99)]

        retriever.retrieve.return_value = contexts
        reader.extract.return_value = expected_answers

        pipeline = ExtractiveQAPipeline(retriever=retriever, reader=reader)
        result = pipeline.run("What is the capital of France?", top_k=3)

        retriever.retrieve.assert_called_once_with(query="What is the capital of France?", k=3)
        reader.extract.assert_called_once_with(
            question="What is the capital of France?",
            contexts=contexts,
            include_contexts=False,
        )
        assert result == expected_answers

    def test_run_passes_include_contexts(self):
        retriever = MagicMock()
        reader = MagicMock()
        retriever.retrieve.return_value = []
        reader.extract.return_value = []

        pipeline = ExtractiveQAPipeline(retriever=retriever, reader=reader)
        pipeline.run("question?", include_contexts=True)

        _, kwargs = reader.extract.call_args
        assert kwargs["include_contexts"] is True
