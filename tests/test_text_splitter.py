from rag.ingestion.text_splitter import TextSplitter


def test_text_splitter_basic_split():
    splitter = TextSplitter(chunk_size=10, chunk_overlap=2)
    text = 'abcdefghijklmno'
    chunks = splitter.split_text(text)
    assert len(chunks) > 0
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_text_splitter_empty_text():
    splitter = TextSplitter(chunk_size=10, chunk_overlap=2)
    assert splitter.split_text('') == []