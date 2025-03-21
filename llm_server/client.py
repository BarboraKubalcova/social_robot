import requests
import time

prompts = [
    "Ako sa máš",
    "Ako sa voláš",
    "Aký je rozdiel medzi jing a jang",
    "Povedz mi v troch vetách niečo o umelej inteligencii",
    "Byť či nebyť",
    "Študovať na KKUI je veľmi náročné"
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
    start_time = time.time()

    for prompt in prompts:
        response = get_llm_response(prompt)
        print("LLM Response:", response)
    
    print(f"Time of responce ({len(prompts)} prompts): {(time.time() - start_time):.4f} sec")
