from rag.vectorstore.chroma_store import ChromaVectorStore


def test_vectorstore_add_and_count(tmp_path):
    store = ChromaVectorStore(persist_dir=tmp_path, collection_name='test_collection')
    result = store.add_documents([
        {'chunk_id': 'c1', 'text': 'alpha beta', 'source': 'file1.pdf', 'page': 1},
    ], [[0.1, 0.2, 0.3]])
    assert result['added'] == 1
    assert store.count() == 1