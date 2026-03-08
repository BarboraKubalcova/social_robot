import os
from typing import Dict, List, Tuple, Optional
from langchain_chroma import Chroma
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from execution.rag.get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma_db/"
DEFAULT_RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "1.4"))
DEFAULT_RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# CONVERSATION HISTORY (for reference resolution only):
# {history}
RAG_PROMPT_TEMPLATE = """
You are an assistant with access to a knowledge base.

Use the DATABASE CONTEXT to answer the user's question.
Use the conversation history ONLY to resolve references (e.g., "that", "it", "the appointment we discussed").
Do NOT answer old questions from the history unless the user explicitly asks again.

DATABASE CONTEXT:
{context}

Question: {question}

Answer:
"""

NO_CONTEXT_PROMPT_TEMPLATE = """
You are an assistant with access to a document database.

The similarity search did not find any sufficiently relevant documents for this question.
1) Say explicitly that the requested information is not present in the database.
2) Then answer using general knowledge, if possible.
3) If you don't know, say you don't know.


Question: {question}

Answer:
"""

class Retriever:
    def __init__(self):
        self.embedding_function = get_embedding_function()
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_function)

    def retrieve_and_build_prompt(self, query_text: str, history_text: str = "") -> Tuple[ChatPromptTemplate, dict, str]:
        """
        Search for documents and build the appropriate prompt.
        Returns: (prompt_template, prompt_kwargs, mode)
        """
        results = self.db.similarity_search_with_score(query_text, k=DEFAULT_RAG_TOP_K)
        
        # Debug logging
        if results:
            best_doc, best_score = results[0]
            print(f"[RAG DEBUG] Query: '{query_text}'")
            print(
                f"[RAG DEBUG] Best score: {best_score:.4f}, "
                f"Threshold: {DEFAULT_RAG_SIMILARITY_THRESHOLD:.4f}"
            )
            print(f"[RAG DEBUG] Found {len(results)} results")

        mode, prompt_template, context_text = self._decide_mode(query_text, history_text, results)

        prompt_kwargs = {
            "history": history_text,
            "question": query_text,
        }
        if mode == "rag":
            prompt_kwargs["context"] = context_text

        return prompt_template, prompt_kwargs, mode

    def _decide_mode(self, query_text: str, history_text: str, results: List[Tuple]) -> Tuple[str, ChatPromptTemplate, str]:
        """
        Decide whether to use RAG or pure LLM based on similarity scores.
        """
        similarity_threshold = DEFAULT_RAG_SIMILARITY_THRESHOLD

        if not results:
             return "llm_only", ChatPromptTemplate.from_template(NO_CONTEXT_PROMPT_TEMPLATE), ""

        # Check best score
        best_doc, best_score = results[0]
        print(
            f"[RAG DEBUG] Mode decision: score {best_score:.4f} < "
            f"{similarity_threshold:.4f} => {best_score < similarity_threshold}"
        )

        if best_score < similarity_threshold:
            mode = "rag"
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
            prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
            return mode, prompt_template, context_text
        else:
            mode = "llm_only"
            prompt_template = ChatPromptTemplate.from_template(NO_CONTEXT_PROMPT_TEMPLATE)
            return mode, prompt_template, ""


"""
Poznamky: 
pri AI agentoch nie je dobre mat historiu chatu v prompte, lebo sa to potom rekurzivne cykli 
kvoli tomu, ze v historii su dalsie otazky a odpovede, ktore sa potom znovu posielaju do RAGu a ten ich zase vracia do promptu atd.

"""