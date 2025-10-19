from langchain.agents import initialize_agent, Tool, AgentType
from langchain_ollama import OllamaLLM
import requests

BASE_URL = "http://127.0.0.1:8000"

# ---- Model ----
model = OllamaLLM(model="mistral")

# ---- Prompt ----
SYSTEM_PROMPT = """
You are SliderBot — a friendly assistant who can talk and control a remote servo slider via API.

You have two tools:
1. get_value() → returns the current slider value (0–180)
2. set_value(value) → sets the slider to a new integer between 0 and 180

Instructions:
- Use get_value when the user asks for the current position.
- Use set_value when the user asks to move or change the slider.
- Speak naturally. After using a tool, describe the result briefly.
- For anything else, just chat normally.
"""

# ---- Tool functions ----
def get_value(_: str = "") -> str:
    try:
        r = requests.get(f"{BASE_URL}/get_value")
        return f"The slider is currently set to {r.json()['value']}."
    except Exception as e:
        return f"Error getting value: {e}"

def set_value(value: str) -> str:
    try:
        val = int(value)
        r = requests.post(f"{BASE_URL}/set_value", json={"value": val})
        return f"The slider has been set to {r.json()['value']}."
    except Exception as e:
        return f"Error setting value: {e}"

# ---- Tools ----
tools = [
    Tool(
        name="get_value",
        func=get_value,
        description="Get the current position of the slider. Input is ignored."
    ),
    Tool(
        name="set_value",
        func=set_value,
        description="Set the slider position to a value between 0 and 180. Input must be a number as a string."
    ),
]

# ---- Agent ----
agent = initialize_agent(
    tools=tools,
    llm=model,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False,
    agent_kwargs={"system_prompt": SYSTEM_PROMPT},
    handle_parsing_errors=True
)

# ---- Chat Loop ----
if __name__ == "__main__":
    print("🤖 SliderBot ready!")
    print("Examples:")
    print("- what’s the slider value?")
    print("- set it to 120")
    print("- exit to quit\n")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            response = agent.invoke({"input": user_input})
            output = response.get("output") or response.get("output_text") or str(response)
            print(f"Agent: {output}")
        except Exception as e:
            print(f"⚠️ Error: {e}")
