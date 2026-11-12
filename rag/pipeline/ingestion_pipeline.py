from __future__ import annotations

from pathlib import Path

from rag.config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR, CHUNK_OVERLAP, CHUNK_SIZE
from rag.embeddings.embedding_model import EmbeddingModel
from rag.ingestion.document_loader import DocumentLoader
from rag.ingestion.text_cleaner import TextCleaner
from rag.ingestion.text_splitter import TextSplitter
from rag.vectorstore.chroma_store import ChromaVectorStore


class IngestionPipeline:
    """Process a PDF into cleaned chunks, embeddings, and vector store entries."""

    def __init__(self):
        self.loader = DocumentLoader()
        self.cleaner = TextCleaner()
        self.splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.embedding_model = EmbeddingModel()
        self.vector_store = ChromaVectorStore(persist_dir=CHROMA_DB_DIR, collection_name=CHROMA_COLLECTION_NAME)

    def process_file(self, file_path: str | Path):
        source_name = Path(file_path).name
        raw_text = self.loader.load(file_path)
        cleaned_text = self.cleaner.clean(raw_text)
        chunks = self.splitter.split_text(cleaned_text)

        chunk_payload = []
        for idx, chunk in enumerate(chunks):
            chunk_payload.append({
                'chunk_id': f'{source_name}-{idx}',
                'text': chunk,
                'source': source_name,
                'page': idx + 1,
            })

        embeddings = self.embedding_model.embed_documents([item['text'] for item in chunk_payload])
        self.vector_store.add_documents(chunk_payload, embeddings)
        return {'file': source_name, 'chunks_count': len(chunk_payload)}