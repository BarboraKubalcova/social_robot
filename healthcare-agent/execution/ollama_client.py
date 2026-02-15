from langchain_ollama import OllamaLLM

class OllamaClient:
    def __init__(self):
        # Initialize the LangChain Ollama wrapper
        self.llm = OllamaLLM(model="qwen3:8b")

    def generate(self, prompt: str):
        """
        Generate a response using the LangChain Ollama wrapper.
        """
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return "Error: Could not communicate with LLM."
    