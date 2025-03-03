import ollama
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForMaskedLM
from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator
import torch 

app = Flask(__name__)



def translate_text(text: str, source_lang = "slovak", target_lang = "english") -> str:
    translator = GoogleTranslator(
        source=source_lang, 
        target=target_lang
    )
    try:
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        return f"Translation error: {e}"
    
def use_mistral(prompt):
    model_name = "mistral" 
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.get("message", {}).get("content", "")

def use_slovak_gpt(prompt):
    tokenizer = AutoTokenizer.from_pretrained("Milos/slovak-gpt-j-1.4B")
    model = AutoModelForCausalLM.from_pretrained("Milos/slovak-gpt-j-1.4B").to("cuda" if torch.cuda.is_available() else "cpu")
    
    encoded_input = tokenizer(prompt, return_tensors='pt').to(model.device)
    # output = model.generate(**encoded_input, max_length=50)
    # Generate text with sampling
    output = model.generate(
        **encoded_input,
        max_length=100,         # Increase length for better responses
        do_sample=True,         # Enable sampling for diversity
        temperature=0.7,        # Controls randomness (0.7-1.0 works well)
        top_k=50,               # Consider only the top 50 words
        top_p=0.9,              # Nucleus sampling (keep top 90% probability mass)
        repetition_penalty=1.2, # Penalize repeating phrases
        no_repeat_ngram_size=3, # Prevents repeating 3-word sequences
        early_stopping=True     # Stop if EOS token is generated
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def use_slovak_bert(prompt):
    model_name = "gerulata/slovakbert"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Create a fill-mask pipeline
    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    # Ensure the correct mask token is used
    mask_token = tokenizer.mask_token  # This is "<mask>"
    if mask_token not in prompt:
        return f"Error: Your prompt must include {mask_token}."

    # Get predictions
    predictions = fill_mask(prompt)

    # Return the top predicted words
    return [pred["token_str"] for pred in predictions]


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        print(f"Original input prompt: {prompt}")
        prompt = translate_text(prompt, 
                source_lang="slovak", 
                target_lang="english"
        )
        print(f"Translated prompt sk->eng: {prompt}")

        generated_text = use_mistral(prompt)
        
        print(f"Original generated text: {generated_text}")
        generated_text = translate_text(generated_text, 
            source_lang="english", 
            target_lang="slovak"
        )
        print(f"Translated generated text eng->sk: {generated_text}")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"generated_text": generated_text})

if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5000, debug=True)  # Listen on all network interfaces
    # ans = use_slovak_gpt("Povedz mi vtip.")
    ans = use_slovak_bert("Toto je <mask> deň.")
    print(ans)

# response = ollama.chat(model=model, messages=[
# {
# 'role': 'user',
# 'content': 'Ako sa máš?',
# },])
# print(response['message']['content'])
