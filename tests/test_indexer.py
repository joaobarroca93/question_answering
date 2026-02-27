from unittest.mock import MagicMock, call

from src.indexer.database_indexer import DatabaseIndexer
from src.entities.document import Document


def make_doc(doc_id: str, content: str) -> Document:
    return Document(id=doc_id, content=content, length=len(content))


class TestDatabaseIndexer:
    def test_index_without_encoder_calls_add_documents(self):
        client = MagicMock()
        indexer = DatabaseIndexer(client=client)
        docs = [make_doc("1", "hello"), make_doc("2", "world")]

        indexer.index(docs)

        client.add_documents.assert_called_once_with(docs)

    def test_index_with_encoder_sets_vectors(self):
        client = MagicMock()
        encoder = MagicMock()
        encoder.batch_encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        indexer = DatabaseIndexer(client=client, encoder=encoder)
        docs = [make_doc("1", "hello"), make_doc("2", "world")]

        indexer.index(docs)

        assert docs[0].vector == [0.1, 0.2]
        assert docs[1].vector == [0.3, 0.4]
        encoder.batch_encode.assert_called_once_with(texts=["hello", "world"])
        client.add_documents.assert_called_once_with(docs)

    def test_index_with_encoder_does_not_mutate_unindexed_docs(self):
        client = MagicMock()
        encoder = MagicMock()
        encoder.batch_encode.return_value = [[1.0]]
        indexer = DatabaseIndexer(client=client, encoder=encoder)

        other_doc = make_doc("other", "untouched")
        doc = make_doc("1", "hello")
        indexer.index([doc])

        assert other_doc.vector is None
