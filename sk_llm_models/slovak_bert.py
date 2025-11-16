from transformers import AutoTokenizer, pipeline, AutoModelForMaskedLM

def use_slovak_bert(prompt, num_predictions = 5):
    model_name = "gerulata/slovakbert"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    mask_token = tokenizer.mask_token  # "<mask>"
    if mask_token not in prompt:
        return f"Error: Your prompt must include {mask_token}."

    predictions = fill_mask(prompt, top_k = num_predictions)
    return [pred["token_str"] for pred in predictions]


def fill_last_word(prompt, n_preds=5):
    model_name = "gerulata/slovakbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mask_token = tokenizer.mask_token
    
    last_word = prompt.split(" ")[-1]
    prompt_with_mask = " ".join(prompt.split(' ')[:-1] + [mask_token])
    predictions = use_slovak_bert(prompt_with_mask, num_predictions=n_preds)
    
    fill = None
    for word in predictions:
        lw_len = len(last_word)
        word = word.strip()
        if last_word == word[:lw_len]:
            fill = word 
            break
    if not fill:
        fill = last_word
    else:
        print(f"Predictions from slovakBERT: {predictions}")
    new_prompt = " ".join(prompt.split(' ')[:-1] + [fill])
    return (new_prompt, predictions)