import pytest
from unittest.mock import MagicMock

from src.repositories.documents_repository import CsvDocumentsRepository, DatasetDocumentsRepository
from src.entities.document import Document


class TestDatasetDocumentsRepository:
    def _make_split(self, rows):
        return rows

    def test_get_all_deduplicates_by_id(self):
        rows = [
            {"document_id": "id1", "document": "content A"},
            {"document_id": "id1", "document": "content A"},  # duplicate
            {"document_id": "id2", "document": "content B"},
        ]
        dataset = MagicMock()
        dataset.values.return_value = [rows]

        repo = DatasetDocumentsRepository(dataset=dataset)
        result = repo.get_all()

        assert len(result) == 2
        ids = {d.id for d in result}
        assert ids == {"id1", "id2"}

    def test_get_all_sets_length(self):
        rows = [{"document_id": "id1", "document": "hello"}]
        dataset = MagicMock()
        dataset.values.return_value = [rows]

        repo = DatasetDocumentsRepository(dataset=dataset)
        result = repo.get_all()

        assert result[0].length == len("hello")

    def test_get_all_across_splits(self):
        split1 = [{"document_id": "id1", "document": "content A"}]
        split2 = [{"document_id": "id2", "document": "content B"}]
        dataset = MagicMock()
        dataset.values.return_value = [split1, split2]

        repo = DatasetDocumentsRepository(dataset=dataset)
        result = repo.get_all()

        assert len(result) == 2

    def test_get_all_cross_split_deduplication(self):
        split1 = [{"document_id": "shared", "document": "content"}]
        split2 = [{"document_id": "shared", "document": "content"}]
        dataset = MagicMock()
        dataset.values.return_value = [split1, split2]

        repo = DatasetDocumentsRepository(dataset=dataset)
        result = repo.get_all()

        assert len(result) == 1

    def test_custom_column_names(self):
        rows = [{"my_id": "x1", "my_content": "some text"}]
        dataset = MagicMock()
        dataset.values.return_value = [rows]

        repo = DatasetDocumentsRepository(
            dataset=dataset,
            document_id_column="my_id",
            document_content_column="my_content",
        )
        result = repo.get_all()

        assert len(result) == 1
        assert result[0].id == "x1"
        assert result[0].content == "some text"
