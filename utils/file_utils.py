from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def validate_pdf_file(file_path: str | Path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    if path.suffix.lower() != '.pdf':
        raise ValueError('Only PDF files are supported.')
    return True


def save_uploaded_file(file_obj, destination: str | Path):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, 'wb') as output_file:
        output_file.write(file_obj.read())
    return destination