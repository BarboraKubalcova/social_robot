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

prompts_mml = [
    "Čo sú to eigenvalues a ako sa počítajú?",
    "Ako funguje algoritmus k-means?",
    "Čo je to gradient descent?",
    "Aký je rozdiel medzi supervised a unsupervised learning?",
    "Ako funguje PCA?",
]

def get_response(prompt):
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
    while True:
        prompt = input("\033[92m"+ "Študent: " + "\033[0m")
        start_time = time.time()
        response = get_response(prompt)
        print("\033[91m"+ "AI: " + "\033[0m" + response)
        print(f"Odpoveď trvala {(time.time() - start_time):.4f} sekúnd")

