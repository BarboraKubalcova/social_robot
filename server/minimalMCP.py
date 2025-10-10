from fastapi import FastAPI
from pydantic import BaseModel
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
PROMPT_TEMPLATE = """
Previous conversation: 
{history}

Answer the question based only on the following context:

{context}

---

Answer the question based on the above context and previous conversation: {question}
"""

# LLM call
model = OllamaLLM(model="mistral")


# ---------------------------
# Embedding Function
# ---------------------------
def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


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

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(history=history_text, context=context_text, question=query_text)

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _ in results]

    # Save the new turn
    history.append((query_text, response_text))
    sessions[session_id] = history
    prev_conv = history_text
    return response_text, sources, prev_conv


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
    sources: list[str]


@app.post("/search_docs", response_model=QueryResponse)
def search_docs(request: QueryRequest):
    answer, sources, prev_text = query_rag(request.query, request.session_id)
    return QueryResponse(answer=answer, sources=sources, history=prev_text)


# ---------------------------
# Optional: Local Test
# ---------------------------
if __name__ == "__main__":
    print("Starting MCP server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
