import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import shutil

# Mock the heavy SentenceTransformer before it's imported by our application code.
# This makes tests run instantly without needing to download a real model.
mock_embedder = MagicMock()
# Configure the mock to return a dummy vector of the correct shape and type.
mock_embedder.encode.return_value = np.random.rand(1, 384).astype('float32')

# The patch must target where the object is *used*, not where it's defined.
# We replace the entire sentence_transformers library in memory before our code loads it.
module_patch = patch.dict('sys.modules', {
    'sentence_transformers': MagicMock(SentenceTransformer=MagicMock(return_value=mock_embedder))
})
module_patch.start()

# Now that the mock is in place, we can safely import our application's modules.
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.storage.vector_store import VectorStore
from src.utils import config


class TestPipelines(unittest.TestCase):
    """Basic tests for the ingestion and retrieval pipelines."""

    def setUp(self):
        """Set up a temporary directory, a dummy transcript, and a vector store for each test."""
        self.test_dir = "temp_pipeline_test"
        os.makedirs(self.test_dir, exist_ok=True)

        self.test_file_path = os.path.join(self.test_dir, "pipeline_test_transcript.txt")
        with open(self.test_file_path, "w", encoding="utf-8") as f:
            f.write("[00:01] Test Speaker: This is a test for the pipeline.")

        test_index_path = os.path.join(self.test_dir, "pipeline_test.faiss")
        test_metadata_path = os.path.join(self.test_dir, "pipeline_test.pkl")

        # Create a single vector store instance to be shared, just like in the real app.
        # We pass the embedding dimension directly to ensure stability.
        self.vector_store = VectorStore(
            index_path=test_index_path,
            metadata_path=test_metadata_path
        )

    def tearDown(self):
        """Clean up the temporary directory and all its contents after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @classmethod
    def tearDownClass(cls):
        """Stop the module patch after all tests in this class are done."""
        module_patch.stop()

    def test_ingestion_and_retrieval_integration(self):
        """
        Tests that the ingestion pipeline correctly adds data that the
        retrieval pipeline can then find.
        """
        # 1. Ingestion Phase
        ingestion_pipeline = IngestionPipeline(self.vector_store)
        ingestion_pipeline.run(self.test_file_path, "call_pipeline_test")

        # Assert that the mock embedder was called during ingestion.
        mock_embedder.encode.assert_called()

        # Assert that the document was successfully added to the store.
        self.assertEqual(self.vector_store.index.ntotal, 1)
        self.assertEqual(self.vector_store.metadata[0]['text'], "This is a test for the pipeline.")

        # 2. Retrieval Phase
        retrieval_pipeline = RetrievalPipeline(self.vector_store)
        query_text = "test the pipeline"

        results = retrieval_pipeline.retrieve_relevant_docs(query_text)

        # Assert that the mock embedder was called again with the query text.
        mock_embedder.encode.assert_called_with([query_text], convert_to_numpy=True)

        # Assert that we got the correct document back.
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['text'], "This is a test for the pipeline.")