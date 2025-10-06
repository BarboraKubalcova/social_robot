import sys
sys.path.insert(0, 'llm_models')
sys.path.insert(0, 'rag')

from flask import Flask, request, jsonify
import mistral
import translation as tr
import slovak_bert as sb
import query_data as qd
from langchain_community.llms.ollama import Ollama

model = Ollama(model="mistral")
app = Flask(__name__)

def fill(prompt):
    filled_prompt, _ = sb.fill_last_word(prompt, n_preds=20)
    print("\033[91m"+ f"Original prompt: {prompt}" + "\033[0m")
    print(f"Filled prompt: {filled_prompt}")
    return filled_prompt

def trasnlate(prompt, from_language, to_language):
    prompt = tr.translate_text(prompt, 
            source_lang=from_language, 
            target_lang=to_language
    )
    print(f"Translated prompt {from_language}->{to_language}: {prompt}")
    return prompt


def use_llm(prompt):
    # mozno sa neskor rozsiri funkcionalita    
    generated_text = mistral.generate(prompt)
    print(f"Original generated text: {generated_text}")
    return generated_text


def use_rag(question):
    # mozno sa neskor rozsiri funkcionalita
    res, sources = qd.query_rag(question, model)
    print(f"Question: {question}\nAnswer: {res}")
    return res


def choose_model(query_text):
    # rozhodovanie o pouzivani LLM alebo RAG modelu
    # ak sa otazka zaobera zaobera vybranou domenou, tak sa pouzije RAG, 
    # ak sa pouzivatel bude chciet len rozpravat, tak sa pouzije LLM  
    use_rag = qd.is_answer_in_database(query_text)
    if use_rag:
        print("\033[92m" + "Ansvering with RAG" + "\033[0m")
        return "RAG"
    else:
        print("\033[92m" + "Ansvering with LLM" + "\033[0m")
        return "LLM"
    

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        prompt = fill(prompt)  # Fill the last word of the prompt
        prompt = trasnlate(prompt, "slovak", "english")  
        generated_text = use_rag(prompt) if choose_model(prompt) == "RAG" else use_llm(prompt)
        generated_text = trasnlate(generated_text, "english", "slovak")
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"generated_text": generated_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # Listen on all network interfaces 
    # ans = use_slovak_llm("Prečo je nebo modré?")
    # ans = use_slovak_bert("Toto je <mask> deň.")
    # print(ans)
    


# response = ollama.chat(model=model, messages=[
# {
# 'role': 'user',
# 'content': 'Ako sa máš?',
# },])
# print(response['message']['content'])
