import os

from langchain_ollama import ChatOllama

class OllamaClient:
    def __init__(self):
        # Initialize the LangChain Ollama chat wrapper
        model_name = os.getenv("OLLAMA_MODEL", "qwen3:8b")
        temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
        # num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
        self.retry_num_predict = int(os.getenv("OLLAMA_RETRY_NUM_PREDICT", "512"))
        self.model_name = model_name
        self.temperature = temperature
        self.num_ctx = num_ctx

        print("Initializing Ollama LLM client")
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            # num_predict=num_predict,
            num_ctx=num_ctx,
        )

    def generate(self, prompt: str):
        """
        Generate a response using the LangChain Ollama wrapper.
        """
        try:
            # print(f"Invoking Ollama with prompt: {prompt}")
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", response)

            # if isinstance(content, list):
            #     text_parts = [
            #         part.get("text", "")
            #         for part in content
            #         if isinstance(part, dict)
            #     ]
            #     content = "\n".join([part for part in text_parts if part]).strip()

            # content = str(content).strip()
            # response_metadata = getattr(response, "response_metadata", {}) or {}
            # done_reason = response_metadata.get("done_reason")

            # if not content and done_reason == "length":
            #     retry_llm = ChatOllama(
            #         model=self.model_name,
            #         temperature=self.temperature,
            #         num_predict=self.retry_num_predict,
            #         num_ctx=self.num_ctx,
            #     )
            #     retry_response = retry_llm.invoke(prompt)
            #     retry_content = getattr(retry_response, "content", retry_response)
            #     if isinstance(retry_content, list):
            #         text_parts = [
            #             part.get("text", "")
            #             for part in retry_content
            #             if isinstance(part, dict)
            #         ]
            #         retry_content = "\n".join([part for part in text_parts if part]).strip()
            #     content = str(retry_content).strip()
            print(f"Ollama response content: {content}")
            if not content:
                return "I’m sorry, I couldn’t generate a response right now. Please try again."
            print(f"Generating response: {content}")
            return content
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return "Error: Could not communicate with LLM."
    