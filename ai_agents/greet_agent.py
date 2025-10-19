from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import OllamaLLM
from langchain_core.runnables.history import RunnableWithMessageHistory

SYSTEM_PROMPT = """
You are a friendly greeting agent. Your only goal is to greet people politely.
If the user tells you their name, remember it and greet them personally.
Once the greeting is done, politely confirm that the conversation can end.
Do not loop or continue indefinitely.
"""

# 1. Model
model = OllamaLLM(model="mistral")

# 2. Define tool
def say_hello(name: str):
    message = f"Hello {name}!"
    print(message)
    return message

tools = [
    Tool(
        name="greet",
        func=say_hello,
        description="Use this tool to greet people by their name. Returns a greeting as text."
    )
]

# 3. Create chat history
message_history = ChatMessageHistory()

# 4. Initialize simpler agent (non-looping)
agent = initialize_agent(
    tools=tools,
    llm=model,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # simpler, no endless reasoning
    verbose=True,
    agent_kwargs={"system_prompt": SYSTEM_PROMPT},
    handle_parsing_errors=True
)

# 5. Wrap the agent for modern memory use
agent_with_memory = RunnableWithMessageHistory(
    agent,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# 6. Helper to preserve history context
def run_with_history(agent_with_memory, user_input, message_history):
    past = "\n".join([f"{m.type}: {m.content}" for m in message_history.messages])
    prompt = f"The conversation so far:\n{past}\nUser: {user_input}"

    response = agent_with_memory.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": "default"}}
    )
    return response

# 7. Run loop
print("Hi, introduce yourself:")

while True:
    user_input = input("> ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Bye!")
        break

    response = run_with_history(agent_with_memory, user_input, message_history)

    # Mistral sometimes returns dicts with either 'output' or 'output_text'
    output = response.get("output") or response.get("output_text") or str(response)
    print(f"Bot: {output}")

    # Optional: stop when the model explicitly says it's done
    if any(stop_word in output.lower() for stop_word in ["done", "complete", "finished"]):
        print("Bot signaled end of conversation.")
        break
