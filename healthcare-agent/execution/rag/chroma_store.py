import chromadb
import os
from chromadb.config import Settings
from typing import List, Dict

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

class ChromaStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR, settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(
            name="procedure_docs",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to the collection."""
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_text: str, n_results: int = 3):
        """Query the collection."""
        # Note: Chroma uses default embedding function if none provided.
        # For production, pass the Ollama embedding function explicitly.
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
