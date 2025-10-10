from deep_translator import GoogleTranslator

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
    