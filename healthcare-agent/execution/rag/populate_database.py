import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from get_embedding_function import get_embedding_function
# from langchain_community.vectorstores import Chroma

from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db"
DATA_PATH = "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    print("CWD:", os.getcwd())
    print("DATA_PATH abs:", os.path.abspath(DATA_PATH))
    print("CHROMA_PATH abs:", os.path.abspath(CHROMA_PATH))

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    print(f"Loaded documents: {len(documents)}")
    if documents:
        print("First doc metadata:", documents[0].metadata)
        print("First doc chars:", len(documents[0].page_content or ""))

    chunks = split_documents(documents)
    print(f"Chunks created: {len(chunks)}")
    if chunks:
        print("First chunk metadata:", chunks[0].metadata)
        print("First chunk chars:", len(chunks[0].page_content or ""))

    add_to_chroma(chunks)



# def load_documents():
#     # Ensure data directory exists
#     if not os.path.exists(DATA_PATH):
#         os.makedirs(DATA_PATH)
#         print(f"Created {DATA_PATH} directory. Please put your PDFs there.")
#         return []
        
#     document_loader = PyPDFDirectoryLoader(DATA_PATH)
#     return document_loader.load()

def load_documents():
    import glob

    print("DATA_PATH exists:", os.path.exists(DATA_PATH))
    print("DATA_PATH contents:", os.listdir(DATA_PATH) if os.path.exists(DATA_PATH) else "N/A")

    pdfs = (glob.glob(os.path.join(DATA_PATH, "*.pdf")) +
            glob.glob(os.path.join(DATA_PATH, "*.PDF")) +
            glob.glob(os.path.join(DATA_PATH, "**", "*.pdf"), recursive=True) +
            glob.glob(os.path.join(DATA_PATH, "**", "*.PDF"), recursive=True))
    pdfs = sorted(set(pdfs))
    print("PDFs found by glob:", pdfs)

    if not pdfs:
        return []

    # Try loading just the first PDF explicitly
    from langchain_community.document_loaders import PyPDFLoader
    test_pdf = pdfs[0]
    print("Testing PyPDFLoader on:", test_pdf)
    docs = PyPDFLoader(test_pdf).load()
    print("Pages loaded from test PDF:", len(docs))
    print("First page chars:", len(docs[0].page_content or "") if docs else 0)

    # If that worked, load the whole directory (recursive, robust)
    from langchain_community.document_loaders import DirectoryLoader
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    # Also load uppercase .PDF if you have them
    loader2 = DirectoryLoader(DATA_PATH, glob="**/*.PDF", loader_cls=PyPDFLoader)
    docs += loader2.load()

    return docs



def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # Chroma is auto-persisting in recent versions, but explicit persist calls might be deprecated or no-op
        # db.persist() 
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
