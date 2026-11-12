from __future__ import annotations

import re
from typing import Iterable


class Reranker:
    """Re-rank retrieved chunks using a CrossEncoder when available and a lexical fallback otherwise."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or 'cross-encoder/stsb-roberta-base'
        self.cross_encoder = None
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(self.model_name, max_length=512)
        except Exception:
            self.cross_encoder = None

    @staticmethod
    def _normalize(text: str) -> list[str]:
        if not text:
            return []
        return re.findall(r"\b[\w'-]+\b", text.lower())

    def _lexical_rerank(self, query: str, results: list[dict], top_k: int = 4):
        query_terms = set(self._normalize(query))
        if not query_terms:
            return results[:top_k]

        ranked = []
        for item in results:
            doc_text = str(item.get('text', ''))
            doc_terms = self._normalize(doc_text)
            doc_term_set = set(doc_terms)
            overlap = len(query_terms & doc_term_set)
            if overlap == 0:
                score = 0.0
            else:
                score = overlap + (sum(1 for term in query_terms if term in doc_terms) / max(len(query_terms), 1))
            item_copy = dict(item)
            item_copy['score'] = float(item.get('score', 0.0)) + float(score)
            ranked.append(item_copy)

        ranked.sort(key=lambda item: item['score'], reverse=True)
        return ranked[:top_k]

    def rerank(self, query: str, results: list[dict], top_k: int = 4):
        if not results:
            return []

        if self.cross_encoder is not None:
            try:
                pairs = [(query, str(item.get('text', ''))) for item in results]
                scores = self.cross_encoder.predict(pairs)
                ranked = []
                for item, score in zip(results, scores):
                    item_copy = dict(item)
                    item_copy['score'] = float(score)
                    ranked.append(item_copy)
                ranked.sort(key=lambda item: item['score'], reverse=True)
                return ranked[:top_k]
            except Exception:
                pass

        return self._lexical_rerank(query, results, top_k=top_k)