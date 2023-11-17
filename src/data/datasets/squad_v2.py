from typing import Dict, List

from src.utils.hash import hash_text

from .base import (
    BaseDatasetBuilder,
    HuggingFaceDatasetMixIn,
    DatasetDict,
    Dataset,
)

from ..data_models import FeatureNames


class SquadV2DatasetBuilder(BaseDatasetBuilder, HuggingFaceDatasetMixIn):
    def __init__(self) -> None:
        self.path = "squad_v2"
        self._raw_dataset = None

    def get_dataset_names(self) -> List[str]:
        return [self.path]

    def make_train_dataset(self) -> Dataset:
        return self.raw_dataset["train"]

    def make_test_dataset(self) -> Dataset:
        return self.raw_dataset["validation"]

    @staticmethod
    def _generate_ids(example: Dict[str, str], columns: List[str]) -> Dict[str, str]:
        return {
            FeatureNames.QUERY_ID.value: hash_text(example[FeatureNames.QUERY.value]),
            FeatureNames.DOCUMENT_ID.value: hash_text(example[FeatureNames.DOCUMENT.value]),
        }

    def make_dataset(self) -> DatasetDict:
        return DatasetDict(
            {
                "train": self.make_train_dataset(),
                "test": self.make_test_dataset(),
            }
        )

    def make_encoder_dataset(self) -> DatasetDict:
        columns_to_rename = {"question": FeatureNames.QUERY.value, "context": FeatureNames.DOCUMENT.value}
        columns_to_remove = ["title", "id", "answers"]
        dataset_dict = self.make_dataset()
        for split, dataset in dataset_dict.items():
            dataset_dict[split] = self._process_dataset(
                dataset=dataset,
                columns_to_rename=columns_to_rename,
                columns_to_remove=columns_to_remove,
            )
        return dataset_dict.map(self._generate_ids)

    def make_documents_dataset(self) -> DatasetDict:
        # TODO: Implement a documents dataset that only contains
        #  `document`` and `document_id``
        return self.make_encoder_dataset()

    @property
    def raw_dataset(self) -> DatasetDict:
        if not self._raw_dataset:
            self._raw_dataset = self.load_from_hub(path=self.path)
        return self._raw_dataset

    def __repr__(self) -> str:
        return "SquadV2DatasetBuilder()"
