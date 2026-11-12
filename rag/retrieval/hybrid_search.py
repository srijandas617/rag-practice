from __future__ import annotations


class HybridSearch:
    """Blend vector and keyword search results."""

    def __init__(self, vector_weight: float = 0.7, keyword_weight: float = 0.3):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    def fuse(self, vector_results: list[dict], keyword_results: list[dict]):
        combined = {}

        for item in vector_results:
            text = item.get('text', '')
            combined[text] = {
                'text': text,
                'metadata': item.get('metadata', {}),
                'score': self.vector_weight * float(item.get('score', 0.0)),
            }

        for item in keyword_results:
            text = item.get('text', '')
            if text in combined:
                combined[text]['score'] += self.keyword_weight * float(item.get('score', 0.0))
            else:
                combined[text] = {
                    'text': text,
                    'metadata': item.get('metadata', {}),
                    'score': self.keyword_weight * float(item.get('score', 0.0)),
                }

        ranked = sorted(combined.values(), key=lambda item: item['score'], reverse=True)
        return ranked