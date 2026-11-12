from rag.embeddings.embedding_model import EmbeddingModel


def test_embedding_model_shapes():
    model = EmbeddingModel(model_name='sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.embed_documents(['hello world', 'rag pipeline'])
    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0