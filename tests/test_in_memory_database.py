import pytest

from src.clients.in_memory_database import InMemoryDatabaseClient
from src.entities.document import Document


def make_doc(doc_id: str = "doc1", content: str = "hello world") -> Document:
    return Document(id=doc_id, content=content, length=len(content))


class TestInMemoryDatabaseClient:
    def test_add_and_get_document(self):
        client = InMemoryDatabaseClient()
        doc = make_doc()
        client.add_document(doc)
        result = client.get_document("doc1")
        assert result == doc

    def test_get_missing_document_returns_none(self):
        client = InMemoryDatabaseClient()
        assert client.get_document("nonexistent") is None

    def test_add_duplicate_raises_with_id_in_message(self):
        client = InMemoryDatabaseClient()
        doc = make_doc("dup")
        client.add_document(doc)
        with pytest.raises(ValueError, match="dup"):
            client.add_document(doc)

    def test_remove_document(self):
        client = InMemoryDatabaseClient()
        doc = make_doc()
        client.add_document(doc)
        client.remove_document("doc1")
        assert client.get_document("doc1") is None

    def test_remove_missing_document_raises_with_id_in_message(self):
        client = InMemoryDatabaseClient()
        with pytest.raises(ValueError, match="ghost"):
            client.remove_document("ghost")

    def test_get_all_documents(self):
        docs = [make_doc(f"id{i}", f"content {i}") for i in range(3)]
        client = InMemoryDatabaseClient(documents=docs)
        result = client.get_all_documents()
        assert len(result) == 3
        assert set(d.id for d in result) == {"id0", "id1", "id2"}

    def test_init_with_documents(self):
        docs = [make_doc("a"), make_doc("b")]
        client = InMemoryDatabaseClient(documents=docs)
        assert client.get_document("a") is not None
        assert client.get_document("b") is not None
