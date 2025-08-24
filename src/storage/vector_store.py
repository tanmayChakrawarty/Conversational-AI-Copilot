import faiss
import pickle
import numpy as np
import os
from typing import List, Dict, Set
from src.utils import config


class VectorStore:
    """
    Manages the storage and retrieval of vector embeddings and their associated metadata.

    This class uses FAISS for efficient similarity searches on vectors and a separate
    pickle file for storing the metadata (original text, speaker, etc.), linking them
    by their index.
    """

    def __init__(self, index_path: str, metadata_path: str):
        """Initializes the VectorStore, loading existing data if available."""
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = {}  # Maps an integer index_id -> chunk_dictionary
        self.next_id = 0
        self.load()

    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Adds new documents and their embeddings to the store.

        Args:
            chunks: A list of chunk dictionaries containing the text and metadata.
            embeddings: A numpy array of the corresponding vector embeddings.
        """
        if self.index is None:
            # This should be initialized in load(), but as a fallback.
            self.index = faiss.IndexFlatL2(config.EMBEDDING_DIMENSION)

        # FAISS requires float32 numpy arrays.
        embeddings_float32 = embeddings.astype('float32')
        self.index.add(embeddings_float32)

        # Store the metadata for each new vector, mapping its FAISS index to the chunk data.
        for i, chunk in enumerate(chunks):
            # The key is the vector's position in the FAISS index.
            self.metadata[self.next_id + i] = chunk

        # Increment the counter for the next batch of additions.
        self.next_id += len(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Searches the vector store for the k most similar documents.

        Args:
            query_embedding: The vector embedding of the user's query.
            k: The number of results to return.

        Returns:
            A list of the top k matching chunk dictionaries.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # FAISS search returns distances and the indices of the nearest vectors.
        query_embedding_float32 = query_embedding.astype('float32')
        distances, indices = self.index.search(query_embedding_float32, k)

        # Retrieve the metadata for the found indices.
        results = []
        for i in indices[0]:
            # -1 is returned by FAISS if there are fewer than k results.
            if i != -1 and i in self.metadata:
                results.append(self.metadata[i])
        return results

    def save(self):
        """Saves the FAISS index and metadata to disk."""
        # Ensure the directory exists.
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        if self.index:
            faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, 'wb') as f:
            # Save both the metadata dictionary and the ID counter.
            pickle.dump((self.metadata, self.next_id), f)
        print(f"Vector store saved to {self.index_path} and {self.metadata_path}")

    def load(self):
        """Loads the FAISS index and metadata from disk if they exist."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print("Loading existing vector store from disk...")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata, self.next_id = pickle.load(f)
            print("Vector store loaded successfully.")
        else:
            print("No existing vector store found. Initializing a new one.")
            # Initialize a new, empty index if files don't exist.
            self.index = faiss.IndexFlatL2(config.EMBEDDING_DIMENSION)
            self.metadata = {}
            self.next_id = 0

    def get_all_call_ids(self) -> Set[str]:
        """Returns a set of all unique call_ids present in the metadata."""
        return set(chunk['call_id'] for chunk in self.metadata.values())

    def get_chunks_by_call_id(self, call_id: str) -> List[Dict]:
        """Retrieves all chunks for a given call_id, sorted by segment order."""
        call_chunks = [
            chunk for chunk in self.metadata.values() if chunk['call_id'] == call_id
        ]
        # Sort by segment_id to ensure chronological order for summarization.
        return sorted(call_chunks, key=lambda x: x['segment_id'])

    def get_full_transcript(self, call_id: str) -> str:
        """
        Retrieves and reconstructs the full transcript for a given call_id.
        """
        call_chunks = self.get_chunks_by_call_id(call_id)
        if not call_chunks:
            return ""

        transcript_lines = []
        for chunk in call_chunks:
            line = f"{chunk.get('timestamp', '')} {chunk.get('speaker', 'Unknown')}: {chunk.get('text', '')}"
            transcript_lines.append(line)

        return "\n".join(transcript_lines)