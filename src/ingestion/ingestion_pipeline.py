from sentence_transformers import SentenceTransformer
from src.storage.vector_store import VectorStore
from src.ingestion.parser import parse_transcript
from src.ingestion.chunker import Chunker
from src.utils import config


class IngestionPipeline:
    """
    Handles the entire ingestion workflow: parsing, chunking, embedding, and storing.
    Its single responsibility is to process raw data and add it to the vector store.
    """

    def __init__(self, vector_store: VectorStore):
        print("Initializing Ingestion Pipeline...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.chunker = Chunker()
        self.store = vector_store
        print("Ingestion Pipeline initialized.")

    def run(self, file_path: str, call_id: str):
        """
        Executes the full ingestion pipeline for a single file.
        """
        # print(f"Processing {call_id}...")
        segments = list(parse_transcript(file_path, call_id))
        chunks = self.chunker.chunk_document(segments)

        if not chunks:
            print(f"Warning: No chunks were generated for {file_path}. Skipping.")
            return

        chunk_texts = [chunk['text'] for chunk in chunks]

        # print(f"Embedding {len(chunk_texts)} chunks for {call_id}...")
        embeddings = self.embedder.encode(
            chunk_texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        self.store.add_documents(chunks, embeddings)