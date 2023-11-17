from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from datasets import (
    load_dataset,
    DatasetDict,
    Dataset,
    Split,
    IterableDataset,
    IterableDatasetDict,
)


SPLIT_TYPE = Optional[Union[str, Split]]
DATASET_TYPE = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]


class BaseDatasetBuilder(ABC):
    @abstractmethod
    def make_dataset(self) -> DatasetDict:
        raise NotImplementedError

    @abstractmethod
    def make_encoder_dataset(self) -> DatasetDict:
        raise NotImplementedError

    @abstractmethod
    def make_documents_dataset(self) -> DatasetDict:
        raise NotImplementedError

    @abstractmethod
    def get_dataset_names(self) -> List[str]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "BaseDatasetBuilder()"

    @staticmethod
    def _process_dataset(dataset: Dataset, columns_to_rename: Dict[str, str], columns_to_remove: List[str]) -> Dataset:
        for old_name, new_name in columns_to_rename.items():
            dataset = dataset.rename_column(old_name, new_name)
        return dataset.remove_columns(columns_to_remove)


class HuggingFaceDatasetMixIn:
    @staticmethod
    def load_from_hub(path: str, split: SPLIT_TYPE = None) -> DATASET_TYPE:
        return load_dataset(path=path, split=split)
