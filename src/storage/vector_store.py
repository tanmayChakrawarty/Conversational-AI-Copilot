import faiss
import pickle
import numpy as np
import os
from typing import List, Dict, Set


class VectorStore:
    """
    Manages storage and retrieval of vector embeddings and metadata.
    The FAISS index is rebuilt in-memory from a persistent metadata file for robustness.
    """

    def __init__(self, index_path: str, metadata_path: str, embedding_dim: int):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = None
        self.next_id = 0
        self.load()

    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        if self.index is None:
            self.index = faiss.IndexFlatL2(int(self.embedding_dim))

        self.index.add(embeddings.astype('float32'))

        for i, chunk in enumerate(chunks):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding['embedding'] = embeddings[i]
            self.metadata[self.next_id + i] = chunk_with_embedding

        self.next_id += len(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0: return []
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        return [self.metadata[i] for i in indices[0] if i != -1 and i in self.metadata]

    def save(self):
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump((self.metadata, self.next_id), f)
        print(f"Vector store metadata saved to {self.metadata_path}")

    def load(self):
        """
        Loads the store by reading the metadata and rebuilding the FAISS index in memory.
        """
        try:
            d = int(self.embedding_dim)
            self.index = faiss.IndexFlatL2(d)
        except (ValueError, TypeError):
            raise TypeError(f"Embedding dimension must be an integer, but got {self.embedding_dim}")

        self.metadata = {}
        self.next_id = 0

        if os.path.exists(self.metadata_path):
            print(f"Loading metadata from {self.metadata_path}...")
            try:
                with open(self.metadata_path, 'rb') as f:
                    self.metadata, self.next_id = pickle.load(f)

                if self.metadata:
                    print("Rebuilding FAISS index from loaded metadata...")
                    all_embeddings = np.array([data['embedding'] for data in self.metadata.values()]).astype('float32')
                    self.index.add(all_embeddings)
                    print(f"Index rebuilt successfully with {self.index.ntotal} vectors.")
                else:
                    print("Metadata file was empty.")
            except Exception as e:
                print(f"Error loading metadata file: {e}. Starting with a new empty store.")
        else:
            print("No existing vector store found. Initializing a new one.")

    def get_all_call_ids(self) -> Set[str]:
        return set(chunk['call_id'] for chunk in self.metadata.values())

    def get_chunks_by_call_id(self, call_id: str) -> List[Dict]:
        call_chunks = [chunk for chunk in self.metadata.values() if chunk['call_id'] == call_id]
        return sorted(call_chunks, key=lambda x: x['segment_id'])

    def get_full_transcript(self, call_id: str) -> str:
        call_chunks = self.get_chunks_by_call_id(call_id)
        if not call_chunks: return ""
        transcript_lines = [f"{c.get('timestamp', '')} {c.get('speaker', 'U')}: {c.get('text', '')}" for c in
                            call_chunks]
        return "\n".join(transcript_lines)