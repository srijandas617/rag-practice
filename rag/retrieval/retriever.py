from __future__ import annotations

try:
    from rag.retrieval.hybrid_search import HybridSearch
except Exception:  # pragma: no cover
    HybridSearch = None

try:
    from rag.retrieval.keyword_search import KeywordSearch
except Exception:  # pragma: no cover
    KeywordSearch = None

try:
    from rag.retrieval.reranker import Reranker
except Exception:  # pragma: no cover
    Reranker = None

from rag.retrieval.vector_search import VectorSearch


class Retriever:
    """Coordinate retrieval logic for the RAG pipeline."""

    def __init__(
        self,
        vector_store,
        embedding_model,
        top_k: int = 4,
        use_hybrid: bool = False,
        use_reranker: bool = False,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.vector_search = VectorSearch(vector_store)
        self.keyword_search = KeywordSearch() if KeywordSearch is not None else None
        self.hybrid_search = HybridSearch() if HybridSearch is not None else None
        self.reranker = Reranker() if use_reranker and Reranker is not None else None

    def retrieve(self, query: str, filters: dict | None = None, top_k: int | None = None):
        top_k = top_k or self.top_k
        query_embedding = self.embedding_model.embed_query(query)
        vector_results = self.vector_search.search(query_embedding=query_embedding, top_k=top_k, where=filters)

        if self.use_hybrid:
            if self.keyword_search is None or self.hybrid_search is None:
                raise ImportError('Hybrid retrieval requires optional search dependencies.')
            documents = [item.get('text', '') for item in vector_results]
            self.keyword_search.index(documents)
            keyword_results = self.keyword_search.search(query, top_k=top_k)
            results = self.hybrid_search.fuse(vector_results, keyword_results)
        else:
            results = vector_results

        if self.use_reranker and self.reranker:
            results = self.reranker.rerank(query, results, top_k=top_k)

        return results[:top_k]