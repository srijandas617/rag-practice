from __future__ import annotations


class TextSplitter:
    """Fixed-size chunking with overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError('chunk_size must be greater than 0')
        if chunk_overlap < 0:
            raise ValueError('chunk_overlap must be non-negative')
        if chunk_overlap >= chunk_size:
            raise ValueError('chunk_overlap must be smaller than chunk_size')
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []

        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = end - self.chunk_overlap
        return chunks