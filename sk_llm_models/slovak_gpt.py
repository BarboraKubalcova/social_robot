from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def use_slovak_gpt(prompt):
    model_name = "Milos/slovak-gpt-j-1.4B"
    # "Milos/slovak-gpt-j-1.4B", "slovak-nlp/mistral-sk-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    
    encoded_input = tokenizer(prompt, return_tensors='pt',  max_length=50).to(model.device)
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
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

