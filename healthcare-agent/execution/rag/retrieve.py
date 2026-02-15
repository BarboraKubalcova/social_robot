from typing import Dict, List, Tuple, Optional
from langchain_chroma import Chroma
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from execution.rag.get_embedding_function import get_embedding_function

CHROMA_PATH = "../../chroma"

RAG_PROMPT_TEMPLATE = """
You are an assistant with access to a knowledge base.

Use the following context from the database to answer the question.
If the context does not contain the answer, say thast you don't know.

Context:
{context}

Previous conversation:
{history}

---
Answer the question based on the above context and previous conversation.

Question: {question}
"""

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

class Retriever:
    def __init__(self):
        self.embedding_function = get_embedding_function()
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_function)

    def retrieve_and_build_prompt(self, query_text: str, history_text: str = "") -> Tuple[ChatPromptTemplate, dict, str]:
        """
        Search for documents and build the appropriate prompt.
        Returns: (prompt_template, prompt_kwargs, mode)
        """
        results = self.db.similarity_search_with_score(query_text, k=5)

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
        similarity_threshold = 1.0 # Adjust as needed

        if not results:
             return "llm_only", ChatPromptTemplate.from_template(NO_CONTEXT_PROMPT_TEMPLATE), ""

        # Check best score
        best_doc, best_score = results[0]
        # print(f"Best score: {best_score}")

        if best_score < similarity_threshold:
            mode = "rag"
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
            prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
            return mode, prompt_template, context_text
        else:
            mode = "llm_only"
            prompt_template = ChatPromptTemplate.from_template(NO_CONTEXT_PROMPT_TEMPLATE)
            return mode, prompt_template, ""
