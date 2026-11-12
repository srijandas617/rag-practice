from __future__ import annotations


class VectorSearch:
    """Semantic retrieval using the vector store."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search(self, query_embedding: list[float], top_k: int = 4, where: dict | None = None):
        result = self.vector_store.query(query_embedding=query_embedding, n_results=top_k, where=where)
        documents = result.get('documents', [[]])[0]
        metadatas = result.get('metadatas', [[]])[0]
        distances = result.get('distances', [[]])[0]

        hits = []
        for idx, document in enumerate(documents):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            score = distances[idx] if idx < len(distances) else 0.0
            hits.append({
                'text': document,
                'metadata': meta,
                'score': score,
            })
        return hits