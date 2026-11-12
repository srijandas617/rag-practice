from __future__ import annotations

from typing import Sequence

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """SentenceTransformer wrapper for document and query embeddings."""

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts if text and text.strip()]
        if not cleaned:
            return []
        embeddings = self.model.encode(cleaned, show_progress_bar=False)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)

    def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise ValueError('Query cannot be empty.')
        embedding = self.model.encode([query.strip()], show_progress_bar=False)[0]
        return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)