from pathlib import Path

from src.repositories import CsvDocumentsRepository
from src.clients import ChromaDatabaseClient
from src.encoders import SentenceTransformersEncoder
from src.indexer import DatabaseIndexer


MAIN_PATH = Path.cwd().resolve().absolute()
DATA_PATH = MAIN_PATH / "data"
MODELS_PATH = MAIN_PATH / "models"
DOCUMENTS_FILEPATH = DATA_PATH / "documents.csv"
ENCODER_MODEL_FILEPATH = MODELS_PATH / "encoders/sentence-transformers/all-mpnet-base-v2-deus"
DOCUMENT_ID_COLUMN = "document_id"
DOCUMENT_CONTENT_COLUMN = "document_content"
PERSIST_PATH = MAIN_PATH / "data/chroma_db"
COLLECTION_NAME = "documents-all-mpnet-base-deus"

documents_repository = CsvDocumentsRepository(
    path=DOCUMENTS_FILEPATH,
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
