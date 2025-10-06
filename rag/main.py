from query_data import query_rag
import requests
import time

# questions = [
#     "What is Retrieval-Augmented Generation (RAG) and how does it enhance AI-Generated Content?",
#     "What are the key limitations of RAG systems?",
#     "How is RAG applied across different AI modalities?",
#     "What are some potential future directions for RAG research?",
#     "What are the different types of retrievers used in RAG?",
#     "What are the four stages of competence in skill development?",
#     "What is the main difference between social robots and service robots?",
#     "How does reinforcement learning contribute to adaptive human–robot interaction?",
#     "What are the advantages of using cloud computing in robotics research?",
#     "What are the main goals of the PhD thesis discussed in the document?"
# ]

questions = [
    "What is Retrieval-Augmented Generation (RAG) and how does it enhance AI-Generated Content?",
    "What was the previous question?"
]

session_id = "session_1"  # shared conversation


def main():
    # for question in questions:
    #     res, sources = query_rag(question)
    #     print(f"\n\nQuestion: {question}\nAnswer: {res}\nSources: {sources}")

    for question in questions:
        print(f"Question: {question}")
        start_time = time.time()
        resp = requests.post(
            "http://127.0.0.1:8000/search_docs",
            json={"query": question, "session_id": session_id}
        )
        print(f"mcp speed: {(time.time() - start_time)}s")
        print(resp.json())


if __name__ == "__main__":
    main()
