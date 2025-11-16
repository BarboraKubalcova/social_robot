import ollama

def generate(prompt):
    model_name = "mistral" 
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.get("message", {}).get("content", "")
