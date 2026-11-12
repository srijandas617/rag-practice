from rag.retrieval.hybrid_search import HybridSearch


def test_hybrid_search_combines_scores():
    vector_results = [{'text': 'vector doc', 'score': 1.0}]
    keyword_results = [{'text': 'keyword doc', 'score': 0.5}]
    fused = HybridSearch().fuse(vector_results, keyword_results)
    assert len(fused) == 2
    assert fused[0]['score'] >= fused[1]['score']