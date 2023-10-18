from enum import Enum, unique


@unique
class FeatureNames(Enum):
    QUERY: str = "query"
    DOCUMENT: str = "document"
    QUERY_ID: str = "query_id"
    DOCUMENT_ID: str = "document_id"
