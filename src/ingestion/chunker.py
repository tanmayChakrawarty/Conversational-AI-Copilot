from typing import List, Dict


class Chunker:
    """
    A simple chunker that formalizes parsed document segments into trackable chunks.

    In our specific pipeline, the parser already divides the transcript into
    meaningful segments based on speaker turns. This class's main responsibility
    is to assign a unique, sequential ID to each of these segments, making them
    official "chunks" for the vector store.
    """

    def chunk_document(self, document_segments: List[Dict]) -> List[Dict]:
        """
        Assigns a unique segment_id to each parsed segment.

        Args:
            document_segments: A list of segment dictionaries produced by the parser. 
                               Each dictionary contains speaker, text, timestamp, etc.

        Returns:
            A list of chunk dictionaries, with each dictionary now including
            a 'segment_id' key to ensure chronological order and uniqueness.
        """
        chunked_documents = []
        for i, segment in enumerate(document_segments):
            # The segment from the parser is already a good logical chunk.
            # We just need to add a unique identifier.
            chunk = segment.copy()
            chunk['segment_id'] = i
            chunked_documents.append(chunk)

        return chunked_documents