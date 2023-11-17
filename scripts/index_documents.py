from pathlib import Path

from src.data.datasets.squad_v2 import SquadV2DatasetBuilder
from src.repositories import DatasetDocumentsRepository
from src.clients import ChromaDatabaseClient
from src.encoders import SentenceTransformersEncoder
from src.indexer import DatabaseIndexer


MAIN_PATH = Path.cwd().resolve().absolute()
ENCODER_MODEL_FILEPATH = "sentence-transformers/all-mpnet-base-v2"
DOCUMENT_ID_COLUMN = "document_id"
DOCUMENT_CONTENT_COLUMN = "document"
PERSIST_PATH = MAIN_PATH / "data/chroma_db"
COLLECTION_NAME = "documents-squad-v2"

# We will index all document in the Squad V2 dataset
dataset_builder = SquadV2DatasetBuilder()
documents_dataset = dataset_builder.make_documents_dataset()

documents_repository = DatasetDocumentsRepository(
    dataset=documents_dataset,
    document_id_column=DOCUMENT_ID_COLUMN,
    document_content_column=DOCUMENT_CONTENT_COLUMN,
)
encoder = SentenceTransformersEncoder(model_filepath=ENCODER_MODEL_FILEPATH)
client = ChromaDatabaseClient(collection_name=COLLECTION_NAME, persist=True, persist_path=str(PERSIST_PATH))
indexer = DatabaseIndexer(client=client, encoder=encoder)


def main():
    print("Reading the documents from repository")
    docs = documents_repository.get_all()
    print(f"{len(docs)} documents loaded.")

    print("Indexing")
    indexer.index(documents=docs)


if __name__ == "__main__":
    main()
