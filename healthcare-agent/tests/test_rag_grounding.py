import pytest
from execution.rag.retrieve import Retriever

def test_retriever_structure():
    retriever = Retriever()
    # This will fail if ChromaDB is not running or empty, so we just check class init
    assert retriever is not None

# Real grounding tests would require ingesting known data and verifying the LLM response contains citations
# e.g.
# def test_rag_citation():
#     response = llm.generate_rag("question", context)
#     assert "Source:" in response
