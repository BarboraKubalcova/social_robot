from query_data import query_rag

questions = [
        "What is Retrieval-Augmented Generation (RAG) and how does it enhance AI-Generated Content?",
        "What are the key limitations of RAG systems?",
        "How is RAG applied across different AI modalities?",
        "What are some potential future directions for RAG research?",
        "What are the different types of retrievers used in RAG?",
        "What are the four stages of competence in skill development?",
        "What is the main difference between social robots and service robots?",
        "How does reinforcement learning contribute to adaptive human–robot interaction?",
        "What are the advantages of using cloud computing in robotics research?",
        "What are the main goals of the PhD thesis discussed in the document?"
    ]


def main():
    for question in questions:
        res, sources = query_rag(question)
        print(f"\n\nQuestion: {question}\nAnswer: {res}\nSources: {sources}")



if __name__ == "__main__":
    main()