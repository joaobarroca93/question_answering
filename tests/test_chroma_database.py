import pytest

from src.clients.chroma_database import ChromaDatabaseClient
from src.entities.document import Document


def test_chroma_document_lifecycle():
    client = ChromaDatabaseClient("test-chroma-document-lifecycle")
    document = Document(
        id="doc-1",
        content="secure dependency regression check",
        length=len("secure dependency regression check"),
        vector=[0.1, 0.2],
        metadata={"source": "test"},
    )

    client.add_document(document)

    loaded = client.get_document(document.id)
    assert loaded is not None
    assert loaded.id == document.id
    assert loaded.content == document.content
    assert loaded.vector is not None
    assert list(loaded.vector) == pytest.approx([0.1, 0.2])
    assert loaded.metadata == document.metadata

    matches = client.query([[0.1, 0.2]], k=1)
    assert matches[0][0][0].id == document.id

    client.remove_document(document.id)
    assert client.get_document(document.id) is None
