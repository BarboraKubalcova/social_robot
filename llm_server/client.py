import requests
import time

prompts_llm = [
    "Ako sa máš",
    "Ako sa voláš",
    "Aký je rozdiel medzi jing a jang",
    "Povedz mi v troch vetách niečo o umelej inteligencii",
    "Byť či nebyť",
    "Študovať na KKUI je veľmi náročné"
]

prompts_rag = [
    "Aké sú štyri štádiá kompetencie pri rozvoji zručností?",
    "Aký je hlavný rozdiel medzi sociálnymi robotmi a servisnými robotmi?",
    "Ako prispieva učenie sa s posilnením k adaptívnej interakcii medzi človekom a robotom?",
    "Aké sú výhody používania cloud computingu vo výskume robotiky?",
    "Aké sú hlavné ciele doktorandskej práce diskutovanej v dokumente?",
    "Aké sú kľúčové komponenty systému RAG?"
]


def get_llm_response(prompt):
    url = "http://localhost:5000/generate"
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result.get("generated_text", "No response from LLM")
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


if __name__ == "__main__":
    # prompt = input("Enter your prompt: ")

    # start_time = time.time()
    # for prompt in prompts_llm:
    #     response = get_llm_response(prompt)
    #     print("LLM Response:", response)
    # print(f"Time of responce ({len(prompts_llm)} prompts): {(time.time() - start_time):.4f} sec")]
    prompt = prompts_rag[1]
    response = get_llm_response(prompt)
    print("LLM Response:", response)

