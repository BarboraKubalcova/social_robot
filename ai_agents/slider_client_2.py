from langchain_ollama import OllamaLLM
import requests
import re

BASE_URL = "http://127.0.0.1:8000"

# --- Model ---
model = OllamaLLM(model="mistral")

# --- Helper functions for slider ---
def get_value():
    try:
        r = requests.get(f"{BASE_URL}/get_value")
        val = r.json()["value"]
        return f"The slider is currently set to {val}."
    except Exception as e:
        return f"Error getting value: {e}"

def set_value(value: int):
    try:
        r = requests.post(f"{BASE_URL}/set_value", json={"value": value})
        new_val = r.json()["value"]
        return f"The slider has been set to {new_val}."
    except Exception as e:
        return f"Error setting value: {e}"

# --- Prompt helper ---
SYSTEM_PROMPT = """
You are SliderBot, a conversational assistant that can chat and control a slider via API.

If the user asks:
- about the slider position → respond with GET.
- to move, change, or set the slider → respond with SET <number>.
- otherwise → respond with CHAT.

Respond only with one of these three forms:
GET
SET <number>
CHAT <your reply>
"""

def interpret_command(user_input: str):
    """Ask the model what to do."""
    prompt = SYSTEM_PROMPT + f"\nUser: {user_input}\nAssistant:"
    raw = model.invoke(prompt)
    text = raw.strip()
    return text

# --- Main loop ---
if __name__ == "__main__":
    print("🤖 SliderBot ready (fast mode).")
    print("Ask naturally, e.g.:")
    print("- what’s the slider value?")
    print("- set it to 130")
    print("- exit to quit\n")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        intent = interpret_command(user_input)

        # Simple intent routing
        if intent.startswith("GET"):
            print("Agent:", get_value())

        elif intent.startswith("SET"):
            m = re.search(r"\d+", intent)
            if m:
                value = int(m.group())
                print("Agent:", set_value(value))
            else:
                print("Agent: I couldn’t find a number to set.")

        elif intent.startswith("CHAT"):
            print("Agent:", intent.replace("CHAT", "").strip())

        else:
            print("Agent:", intent)
