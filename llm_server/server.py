# from transformers import RobertaTokenizer, RobertaModel
# tokenizer = RobertaTokenizer.from_pretrained('gerulata/slovakbert')
# model = RobertaModel.from_pretrained('gerulata/slovakbert')
# text = "Povedz mi nieco o deleni zlomkov"
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)

import ollama
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize the Ollama model (Mistral)
model_name = "mistral"  # Assuming "mistral" is the name of the Ollama model

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Get the model's response
    try:
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        generated_text = response.get("message", {}).get("content", "")
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
