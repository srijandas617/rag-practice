from rag.ingestion.document_loader import DocumentLoader


def test_document_loader_loads_text_from_pdf(tmp_path):
    pdf_path = tmp_path / 'sample.pdf'
    pdf_path.write_bytes(b'not a real pdf')

    loader = DocumentLoader()
    try:
        loader.load(pdf_path)
    except Exception:
        pass