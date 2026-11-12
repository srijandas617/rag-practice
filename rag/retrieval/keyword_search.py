from __future__ import annotations

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None


class KeywordSearch:
    """BM25 keyword search over document chunks."""

    def __init__(self, documents: list[str] | None = None):
        self.documents = documents or []
        self.tokenized_docs = []
        self.bm25 = None

    def _tokenize(self, text: str):
        return text.lower().split()

    def index(self, documents: list[str]):
        if BM25Okapi is None:
            raise ImportError('rank-bm25 is required for keyword search. Install it with `pip install rank-bm25`.')
        self.documents = documents
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 4):
        if BM25Okapi is None:
            raise ImportError('rank-bm25 is required for keyword search. Install it with `pip install rank-bm25`.')
        if self.bm25 is None:
            return []
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            results.append({
                'text': self.documents[idx],
                'score': float(score),
                'metadata': {'source': 'keyword-search'},
            })
        return results