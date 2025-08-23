from sentence_transformers import SentenceTransformer
from src.storage.vector_store import VectorStore
from src.utils import config
from typing import List, Dict

class RetrievalPipeline:
    """
    Handles the retrieval workflow: embedding a query and searching the vector store.
    Its single responsibility is to find relevant documents for a given query.
    """
    def __init__(self, vector_store: VectorStore):
        print("Initializing Retrieval Pipeline...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.store = vector_store
        print("Retrieval Pipeline initialized.")

    def retrieve_relevant_docs(self, query: str, k: int = 5) -> List[Dict]:
        """
        Embeds a query and retrieves the most relevant document chunks.
        """
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        return self.store.search(query_embedding, k=k)