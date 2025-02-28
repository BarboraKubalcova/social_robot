

import ollama
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator

app = Flask(__name__)

model_name = "mistral" 

def translate_text(text: str, source_lang = "slovak", target_lang = "english") -> str:
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    try:
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        return f"Translation error: {e}"
    

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        print(f"Original input prompt: {prompt}")
        prompt = translate_text(prompt, source_lang="slovak", target_lang="english")
        print(f"Translated prompt sk->eng: {prompt}")
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        generated_text = response.get("message", {}).get("content", "")
        print(f"Original generated text: {generated_text}")
        generated_text = translate_text(generated_text, source_lang="english", target_lang="slovak")
        print(f"Translated generated text eng->sk: {generated_text}")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"generated_text": generated_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # Listen on all network interfaces
    

# response = ollama.chat(model='mistral', messages=[
# {
# 'role': 'user',
# 'content': 'Why is the sky blue?',
# },])

# print(response['message']['content'])
