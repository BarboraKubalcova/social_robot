# Healthcare Agent (Refactored)

A healthcare assistant agent designed to provide:
1.  **Empathetic Chat** (Supportive/Non-clinical)
2.  **RAG QA** (LangChain + ChromaDB over PDFs)
3.  **Action Execution** (Simple Appointment scheduling & Messaging)

## Architecture

The system uses a 3-layer architecture:
1.  **Directives**: Policy and behavior definitions (Markdown).
2.  **Orchestration**: Intent classification and routing (Python).
3.  **Execution**: Deterministic tool execution (Python).

## Setup

1.  **Dependencies**:
    `pip install -e .`

2.  **RAG Setup**:
    -   Place PDFs in `execution/rag/data/`.
    -   Run `python execution/rag/populate_database.py` to index them.

3.  **Run API**:
    `uvicorn api.main:app --reload`
    
4.  **Frontend**:
    Open `web/index.html`.

## Configuration
-   **Ollama**: Expects `qwen3:8b` at `http://localhost:11434`.
-   **Database**: Uses local `healthcare_agent.db` (SQLite) and `chroma_db/` directory.
