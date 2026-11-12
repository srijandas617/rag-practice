import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

UPLOAD_DIR = Path(os.getenv('UPLOAD_DIR', str(BASE_DIR / 'data' / 'uploads')))
CHROMA_DB_DIR = Path(os.getenv('CHROMA_DB_DIR', str(BASE_DIR / 'data' / 'chroma_db')))

CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
TOP_K = int(os.getenv('TOP_K', '4'))
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
CHROMA_COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'production_rag_collection')