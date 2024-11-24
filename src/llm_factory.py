# llm_factory.py
from langchain_ollama import ChatOllama, OllamaLLM

def create_llm(model: str, temperature: float = 0, format: str = None):
    if format == "json":
        return ChatOllama(model=model, temperature=temperature, format="json")
    return ChatOllama(model=model, temperature=temperature)