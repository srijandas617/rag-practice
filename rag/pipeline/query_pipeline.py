from __future__ import annotations

from rag.config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR, TOP_K
from rag.embeddings.embedding_model import EmbeddingModel
from rag.generation.llm import GroqGenerator
from rag.generation.prompt_template import build_prompt
from rag.retrieval.retriever import Retriever
from rag.vectorstore.chroma_store import ChromaVectorStore


class QueryPipeline:
    """Generate contextual answers from ChromaDB retrieval."""

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = ChromaVectorStore(persist_dir=CHROMA_DB_DIR, collection_name=CHROMA_COLLECTION_NAME)
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            top_k=TOP_K,
            use_hybrid=False,
            use_reranker=True,
        )
        self.llm = GroqGenerator()

    def answer(self, question: str, top_k: int = TOP_K):
        results = self.retriever.retrieve(question, top_k=top_k)
        context = '\n\n'.join(item.get('text', '') for item in results if item.get('text'))
        prompt = build_prompt(question, context)
        answer = self.llm.generate(prompt)
        sources = [
            {
                'source': item.get('metadata', {}).get('source', 'unknown'),
                'page': item.get('metadata', {}).get('page', 1),
                'score': item.get('score', 0.0),
            }
            for item in results
        ]
        return {'answer': answer, 'sources': sources, 'context': context}