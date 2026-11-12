from __future__ import annotations

from pathlib import Path

import pymupdf


class DocumentLoader:
    """Load text from PDF files."""

    def __init__(self, file_path: str | Path | None = None):
        self.file_path = Path(file_path) if file_path is not None else None

    def load(self, file_path: str | Path | None = None) -> str:
        path = Path(file_path) if file_path is not None else self.file_path
        if path is None:
            raise ValueError('A file path must be provided.')
        if not path.exists():
            raise FileNotFoundError(f'PDF file not found: {path}')

        pages = []
        with pymupdf.open(str(path)) as document:
            for page_number, page in enumerate(document):
                text = page.get_text('text')
                if text:
                    pages.append(f'\n\n[Page {page_number + 1}]\n{text}')

        return '\n\n'.join(pages)