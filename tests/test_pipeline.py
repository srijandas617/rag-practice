from rag.pipeline.ingestion_pipeline import IngestionPipeline


def test_ingestion_pipeline_initializes():
    pipeline = IngestionPipeline()
    assert pipeline is not None