from query_data import query_rag
from langchain_community.llms.ollama import Ollama
import QnA

EVAL_PROMPT = """
Expected Response: {expected_response}

Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def HRI_script():
    questions = QnA.questions_HRI
    for key, value in questions.items():
        query_and_validate(question=value["Q"], expected_response=value["Ans"])


def RAG_script():
    questions = QnA.questions_RAG
    for key, value in questions.items():
        query_and_validate(question=value["Q"], expected_response=value["Ans"])


def query_and_validate(question: str, expected_response: str):
    response_text, source = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

HRI_script()
RAG_script()
