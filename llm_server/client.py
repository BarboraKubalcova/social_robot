import requests

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
    prompt = input("Enter your prompt: ")
    response = get_llm_response(prompt)
    print("LLM Response:", response)