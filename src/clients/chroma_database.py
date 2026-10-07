import chromadb
from typing import List, Optional, Tuple, cast

from src.entities.document import Document, METADATA, VECTOR

from .base import BaseVectorDatabaseClient


class ChromaDatabaseClient(BaseVectorDatabaseClient):
    def __init__(
        self,
        collection_name: str,
        documents: Optional[List[Document]] = None,
        persist: bool = False,
        persist_path: str = "./chroma",
    ):
        if persist:
            self._chroma_client = chromadb.PersistentClient(path=persist_path)
        else:
            self._chroma_client = chromadb.Client()

        self.collection_name = collection_name
        self.collection = self._chroma_client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        if documents:
            self.add_documents(documents=documents)

    def add_document(self, document: Document) -> None:
        metadatas = [document.metadata] if document.metadata else None
        self.collection.add(
            documents=[document.content],
            embeddings=[cast(VECTOR, document.vector)],
            metadatas=metadatas,
            ids=[document.id],
        )

    def add_documents(self, documents: List[Document]) -> None:
        docs = []
        ids = []
        embeddings = []
        metadatas = []
        for document in documents:
            docs.append(document.content)
            ids.append(document.id)
            embeddings.append(document.vector)
            if document.metadata:
                metadatas.append(document.metadata)

        chroma_metadatas = metadatas if len(metadatas) == len(docs) else None

        self.collection.add(
            documents=docs,
            embeddings=cast(List[VECTOR], embeddings),
            metadatas=chroma_metadatas,
            ids=ids,
        )

    def remove_document(self, document_id: str) -> None:
        self.collection.delete(ids=[document_id])

    def remove_documents(self, document_ids: List[str]) -> None:
        self.collection.delete(ids=document_ids)

    def get_document(self, document_id: str) -> Optional[Document]:
        doc = self.collection.get(ids=[document_id], include=["embeddings", "documents", "metadatas"])
        ids = doc["ids"]
        documents = doc["documents"]
        embeddings = doc["embeddings"]
        metadatas = doc["metadatas"]
        if not ids:
            return None

        content = documents[0] if documents is not None and len(documents) else ""
        return Document(
            id=ids[0],
            content=content,
            length=len(content),
            vector=embeddings[0] if embeddings is not None and len(embeddings) else None,
            metadata=metadatas[0] if metadatas is not None and len(metadatas) else None,
        )

    def get_all_documents(self) -> List[Document]:
        raise NotImplementedError

    def query(
        self,
        query_vectors: List[List[float]],
        k: int = 1,
        metadata: Optional[METADATA] = None,
    ) -> List[Tuple[List[Document], List[float]]]:
        results = self.collection.query(
            query_embeddings=cast(List[VECTOR], query_vectors),
            n_results=k,
            where=dict(metadata) if metadata else None,
            include=["embeddings", "documents", "metadatas", "distances"],
        )
        iterator = zip(
            results["ids"],
            cast(List[List[str]], results["documents"]),
            cast(List[List[METADATA]], results["metadatas"]),
            cast(List[List[VECTOR]], results["embeddings"]),
            cast(List[List[float]], results["distances"]),
        )
        batch_results = []
        for ids, contents, metadatas, vectors, distances in iterator:
            docs = []
            for i in range(len(ids)):
                docs.append(
                    Document(
                        id=ids[i],
                        content=contents[i],
                        length=len(contents[i]),
                        vector=vectors[i],
                        metadata=metadatas[i],
                    )
                )
            batch_results.append((docs, distances))
        return batch_results
