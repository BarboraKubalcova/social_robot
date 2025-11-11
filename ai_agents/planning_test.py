from langchain_ollama import OllamaLLM
import requests
import json
import re

BASE_URL = "http://127.0.0.1:8000"

# --- Model (same as your original) ---
model = OllamaLLM(model="mistral")


# --- Low-level API tools (same behavior as before) ---
def api_get_value():
    try:
        r = requests.get(f"{BASE_URL}/get_value", timeout=10)
        r.raise_for_status()
        return int(r.json()["value"])
    except Exception as e:
        raise RuntimeError(f"GET failed: {e}")


def api_set_value(value: int):
    try:
        r = requests.post(f"{BASE_URL}/set_value", json={"value": int(value)}, timeout=10)
        r.raise_for_status()
        return int(r.json()["value"])
    except Exception as e:
        raise RuntimeError(f"SET failed: {e}")


# --- Planner prompt: produce a tiny JSON plan we can execute deterministically ---
# PLANNER_SYSTEM = """
# You are a careful planner for a slider controller. You do NOT execute actions; you only produce a plan.
#
# Available tools (use only these names):
# - "get_value": no args. Retrieves the current slider integer value.
# - "set_value": args: {"value": <int>}. Sets the slider to an explicit integer.
# - "compute": args: {"op": "increase"|"decrease"|"set", "amount": <int>}.
#    - "increase": add amount to the most recent retrieved value
#    - "decrease": subtract amount from the most recent retrieved value
#    - "set": alias for setting an absolute value; when using "set", also include {"value": <int>} and ignore amount.
#
# Rules:
# - Output STRICT JSON with this schema:
#   {"steps": [{"tool": "...", "args": {...}}, ...]}
# - Prefer a short plan. If the user asks to change by a delta, plan: get_value -> compute -> set_value.
# - Use only tools "get_value", "set_value" and "compute", nothing else.
# - Never include commentary. No markdown. JSON only.
# """


PLANNER_SYSTEM = """
You are a precise planner for a slider control system.

Your job: receive a natural-language request (e.g., “decrease the slider by 2”)
and produce a valid JSON plan that a deterministic executor can follow.

### Available tools (you MUST only use these three names):

1. "get_value" — no arguments. Reads the current integer value of the slider.
2. "compute" — arguments:
   {
     "op": "increase" | "decrease" | "set",
     "amount": <integer>
   }
   Description:
   - "increase": add `amount` to the most recently retrieved value.
   - "decrease": subtract `amount` from the most recently retrieved value.
   - "set": assign an absolute value; when using "set", include "value" instead of "amount".
3. "set_value" — arguments:
   {
     "value": <integer> | "__COMPUTED__"
   }
   Description:
   - Use "__COMPUTED__" when you want to set whatever value was produced by the previous compute step.

### Output requirements:

- You MUST return **strict JSON**, not markdown or text.
- The JSON MUST have exactly this structure:
  {
    "steps": [
      {"tool": "<tool_name>", "args": { ... }},
      ...
    ]
  }
- The JSON MUST contain ONLY the key `"steps"`.
- Do NOT invent any other fields (e.g., "final_value", "result", "metadata", etc.).
- Each tool name MUST be one of: "get_value", "compute", "set_value".
- Arguments MUST follow the schemas listed above.

### Behavioral guidance:

- For requests like "decrease/increase by N":
  [
    {"tool": "get_value", "args": {}},
    {"tool": "compute", "args": {"op": "decrease"|"increase", "amount": N}},
    {"tool": "set_value", "args": {"value": "__COMPUTED__"}}
  ]
- For "set to N":
  [{"tool": "set_value", "args": {"value": N}}]
- For "what is the slider value":
  [{"tool": "get_value", "args": {}}]

Return only valid JSON, nothing else.
"""


def plan(user_input: str) -> dict:
    """Ask the model to create a structured plan (JSON)."""
    user_prompt = f"""Based on the previous rules and tools process this user request: {user_input}

Return ONLY valid JSON with steps. Please bro, Im begging you. Example:
{{
  "steps": [
    {{"tool": "get_value", "args": {{}}}},
    {{"tool": "compute", "args": {{"op": "decrease", "amount": 2}}}},
    {{"tool": "set_value", "args": {{"value": 128}}}}
  ]
}}
"""
    # Compose a simple chat-style prompt
    prompt = PLANNER_SYSTEM + "\n" + user_prompt
    raw = model.invoke(prompt).strip()

    # Robustness: if the model returned extra text, try to extract JSON
    try:
        # Fast path: direct JSON
        print(f"Raw plan: {json.loads(raw)}")
        return json.loads(raw)
    except Exception:
        # Fallback: try to find a JSON object with a naive regex
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            # Final fallback: build a minimal intent-plan from user_input
            print(f"First naive: {naive_backup_plan(user_input)}")
            return naive_backup_plan(user_input)
        try:
            print(f"m group: {m.group(0)}")
            return json.loads(m.group(0))
        except Exception:
            # Final fallback if parsing still fails
            print(f"First naive: {naive_backup_plan(user_input)}")
            return naive_backup_plan(user_input)


def naive_backup_plan(user_input: str) -> dict:
    """Very small heuristic plan if JSON parsing fails."""
    # Try to detect "increase/decrease/set <num>"
    num = re.search(r"-?\d+", user_input)
    if any(k in user_input.lower() for k in ["increase", "up", "raise", "more"]) and num:
        return {"steps": [
            {"tool": "get_value", "args": {}},
            {"tool": "compute", "args": {"op": "increase", "amount": int(num.group())}},
            {"tool": "set_value", "args": {"value": "__COMPUTED__"}}
        ]}
    if any(k in user_input.lower() for k in ["decrease", "down", "lower", "less"]) and num:
        return {"steps": [
            {"tool": "get_value", "args": {}},
            {"tool": "compute", "args": {"op": "decrease", "amount": int(num.group())}},
            {"tool": "set_value", "args": {"value": "__COMPUTED__"}}
        ]}
    if any(k in user_input.lower() for k in ["set", "to", "exact", "value"]) and num:
        return {"steps": [
            {"tool": "set_value", "args": {"value": int(num.group())}}
        ]}
    # Otherwise, just read it
    return {"steps": [{"tool": "get_value", "args": {}}]}


# --- Executor: runs the plan step by step with local state ---
def execute(plan_dict: dict) -> str:
    current_value = None
    computed_value = None

    steps = plan_dict.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return "Plan has no steps."

    for idx, step in enumerate(steps, start=1):
        tool = step.get("tool")
        args = step.get("args", {}) or {}

        try:
            if tool == "get_value":
                current_value = api_get_value()
                # Keep computed in sync with the most recent read
                computed_value = current_value

            elif tool == "compute":
                op = args.get("op")
                amount = int(args.get("amount", 0))
                if current_value is None:
                    # If no prior GET, read first (defensive)
                    current_value = api_get_value()
                if op == "increase":
                    computed_value = current_value + amount
                elif op == "decrease":
                    computed_value = current_value - amount
                elif op == "set":
                    computed_value = int(args.get("value"))
                else:
                    return f"Unknown compute op: {op}"

            elif tool == "set_value":
                # Allow placeholder "__COMPUTED__"
                val = args.get("value")
                if val == "__COMPUTED__":
                    if computed_value is None:
                        return "Nothing computed to set."
                    final = api_set_value(computed_value)
                else:
                    final = api_set_value(int(val))
                current_value = final
                computed_value = final

            else:
                return f"Unknown tool: {tool}"

        except Exception as e:
            return f"Step {idx} ({tool}) failed: {e}"

    return f"Done. Slider is now {current_value}."


# --- Simple chat wrapper: plan -> execute or just chat back ---
def maybe_chat_response(user_input: str) -> str:
    """If user is just chatting, provide a friendly reply."""
    # crude check: if no numbers/verbs, we might just answer
    control_verbs = ["set", "increase", "decrease", "lower", "raise", "value", "get", "read"]
    if not any(v in user_input.lower() for v in control_verbs):
        return ("I can read or change the slider. "
                "Try: 'what’s the slider value', 'set to 130', or 'decrease by 2'.")
    return ""


# --- Main loop ---
if __name__ == "__main__":
    print("🤖 SliderBot (Plan→Execute). Type 'exit' to quit.")
    print("Examples:")
    print("- what's the slider value?")
    print("- set it to 130")
    print("- decrease the value by 2\n")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        # First, see if it’s a casual chat
        chat_hint = maybe_chat_response(user_input)
        # Always attempt planning; casual chat will yield a simple GET or no-op plan
        plan_dict = plan(user_input)

        # If the plan is empty and we had a chat hint, print it
        if not plan_dict.get("steps") and chat_hint:
            print("Agent:", chat_hint)
            continue

        # Execute plan
        result = execute(plan_dict)
        print("Agent:", result)
