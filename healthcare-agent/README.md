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
    `python run_api.py`
    
    This launcher handles `Ctrl+C`, ensures the full Uvicorn process group is stopped,
    and clears stale listeners on port `8000` before startup. To use a different port,
    set `PORT` (example: `PORT=8010 python run_api.py`).
    
4.  **Frontend**:
    Open `web/index.html`.

## Configuration
-   **Ollama**: Expects `qwen3:8b` at `http://localhost:11434`.
-   **Database**: Uses local `healthcare_agent.db` (SQLite) and `chroma_db/` directory.

### Performance tuning (response speed)
-   `ROUTER_MODE`: `agent` (default) or `keyword` (fastest deterministic routing).
-   `MAX_HISTORY_TURNS`: Number of turns injected into prompts (default `3`).
-   `RAG_TOP_K`: Number of retrieved chunks for RAG (default `3`).
-   `OLLAMA_MODEL`: Model name (smaller model = faster).
-   `OLLAMA_NUM_PREDICT`: Max generated tokens (default `120`).
-   `OLLAMA_NUM_CTX`: Context window sent to model (default `2048`).
-   `OLLAMA_TEMPERATURE`: Sampling temperature (default `0.2`).
