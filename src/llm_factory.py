# llm_factory.py
from langchain_ollama import ChatOllama, OllamaLLM

def create_llm(model: str, temperature: float = 0, format: str = None):
    if format == "json":
        return OllamaLLM(model=model, temperature=temperature, format="json", model_kwargs={'device': 'gpu'})
    return OllamaLLM(model=model, temperature=temperature, model_kwargs={'device': 'gpu'})