import uuid
# import pypdf  # Require pypdf installation
from execution.rag.chunking import chunk_text
from execution.rag.chroma_store import ChromaStore

class PDFIngestor:
    def __init__(self):
        self.store = ChromaStore()

    def ingest(self, file_path: str, metadata: dict = {}):
        """
        Ingest a PDF file: extract text, chunk, and store.
        """
        # Placeholder for PDF text extraction since pypdf might not be installed in user env yet
        # text = extract_text_from_pdf(file_path) 
        text = f"Mock text content from {file_path}"
        
        chunks = chunk_text(text)
        
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [metadata.copy() for _ in chunks]
        
        self.store.add_documents(documents=chunks, metadatas=metadatas, ids=ids)
        print(f"Ingested {len(chunks)} chunks from {file_path}")
