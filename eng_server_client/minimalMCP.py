from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Tuple
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import uvicorn

# ---------------------------
# Config
# ---------------------------
conversation_history = []

CHROMA_PATH = "chroma"

# RAG prompt: answer based on context
RAG_PROMPT_TEMPLATE = """
You are an assistant with access to a knowledge base.

Use the following context from the database to answer the question.
If the context does not contain the answer, say that you don't know.

Context:
{context}

Previous conversation:
{history}

---
Answer the question based on the above context and previous conversation.

Question: {question}
"""

# Fallback prompt: no useful context → say it's not in DB, then answer
NO_CONTEXT_PROMPT_TEMPLATE = """
You are an assistant with access to a document database.

For this question, the similarity search did not find any sufficiently relevant documents.
First, explicitly say that the requested information is not present in the database.
Then, answer the question using your general knowledge.
If you don't know the answer, say that you don't know.

Previous conversation:
{history}

Question: {question}
"""

# LLM call
model = OllamaLLM(model="mistral")


# ---------------------------
# Embedding Function
# ---------------------------
def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


# ---------------------------
# Decision + Prompt builder
# ---------------------------
def decide_mode_and_build_prompt(
    query_text: str,
    history_text: str,
    results: List[Tuple],
    similarity_threshold: float = 0.0,
):
    """
    Decide whether to use RAG or pure LLM based on similarity scores.

    - If best_score > similarity_threshold -> use RAG prompt with context.
    - Else -> use NO_CONTEXT prompt, tell user it's not in DB, answer from general knowledge.

    NOTE: This assumes that higher score = more similar.
    If your Chroma setup returns distances (lower = more similar),
    you should invert the condition accordingly.
    """

    if not results:
        # No results at all → pure LLM
        mode = "llm_only"
        prompt_template = ChatPromptTemplate.from_template(NO_CONTEXT_PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            history=history_text,
            question=query_text,
        )
        return mode, prompt

    # Take the best score (first result)
    best_doc, best_score = results[0]
    print(f"Best score: {best_score}")

    # If you are actually getting distances (lower is better),
    # you might want: if best_score < some_distance_threshold instead.
    if best_score < similarity_threshold:
        # Use RAG
        mode = "rag"
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            history=history_text,
            context=context_text,
            question=query_text,
        )
    else:
        # Similarity too weak → treat as not in DB, use general knowledge
        mode = "llm_only"
        prompt_template = ChatPromptTemplate.from_template(NO_CONTEXT_PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            history=history_text,
            question=query_text,
        )

    return mode, prompt


# ---------------------------
# Core RAG Logic
# ---------------------------
# In-memory store
sessions = {}


def query_rag(query_text: str, session_id: str = "default"):
    history = sessions.get(session_id, [])
    history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)

    print("Similarity search results:", results)

    # Decide RAG vs pure LLM and build prompt accordingly
    mode, prompt = decide_mode_and_build_prompt(
        query_text=query_text,
        history_text=history_text,
        results=results,
        similarity_threshold=1.0,  # adjust if needed
    )

    response_text = model.invoke(prompt)

    # Only surface sources if we actually used RAG context
    if mode == "rag":
        sources = [doc.metadata.get("id", None) for doc, _ in results]
    else:
        sources = []

    # Save the new turn
    history.append((query_text, str(response_text)))
    sessions[session_id] = history
    prev_conv = history_text

    return str(response_text), sources, mode


# ---------------------------
# MCP Server Wrapper (FastAPI)
# ---------------------------
app = FastAPI()


# Request/Response schemas
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    mode: Optional[str] = None


@app.post("/search_docs", response_model=QueryResponse)
def search_docs(request: QueryRequest):
    answer, sources, mode = query_rag(request.query, request.session_id)
    return QueryResponse(answer=answer, sources=sources, mode=mode)


# ---------------------------
# Optional: Local Test
# ---------------------------
if __name__ == "__main__":
    print("Starting MCP server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
