from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import chromadb
from chromadb.config import Settings


class ChromaVectorStore:
    """Persistent ChromaDB wrapper for RAG chunks."""

    def __init__(self, persist_dir: str | Path, collection_name: str = 'rag_collection'):
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        self.persist_dir = str(persist_path)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_dir, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_documents(self, chunks: Iterable[dict[str, Any]], embeddings: list[list[float]] | None = None):
        chunks_list = list(chunks)
        if not chunks_list:
            return {'added': 0}
        if embeddings is None:
            raise ValueError('Embeddings are required when adding documents.')
        if len(embeddings) != len(chunks_list):
            raise ValueError('Embedding count must match the number of chunks.')

        ids = [f"{chunk.get('source', 'doc')}-{chunk.get('chunk_id', idx)}" for idx, chunk in enumerate(chunks_list)]
        documents = [chunk.get('text', '') for chunk in chunks_list]
        metadatas = [{
            'source': chunk.get('source', 'unknown'),
            'page': chunk.get('page', 1),
            'chunk_id': chunk.get('chunk_id', idx),
        } for idx, chunk in enumerate(chunks_list)]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return {'added': len(chunks_list)}

    def query(self, query_embedding: list[float], n_results: int = 4, where: dict | None = None):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

    def count(self) -> int:
        return self.collection.count()

    def delete_collection(self):
        self.client.delete_collection(name=self.collection_name)

    def reset(self):
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(name=self.collection_name)