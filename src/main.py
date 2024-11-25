import streamlit as st
import os
import logging
from langchain_ollama import OllamaEmbeddings
from vector_store_lang_chain_impl import VectorStoreLangChainImpl
from control_flow import Controlflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d'
)

# Set logging level for specific libraries to WARNING or ERROR
logging.getLogger("requests").setLevel(logging.WARNING)

os.environ["TAVILY_API_KEY"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["USER_AGENT"] = "PythonLLMProject/1.0"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "local-llama32-rag"

local_llm = "nous-hermes2:10.7b"
vector_store_name = "thermo_texts"
vector_store_lang_chain_impl = VectorStoreLangChainImpl(OllamaEmbeddings(model=local_llm), vector_store_name)

pdf_paths = [
    r"C:\Users\rebel\Documents\textbooks\61-the_principles_of_chemical_equil.pdf",
    r"C:\Users\rebel\OneDrive\Documents\textbooks\repertorium_gibbs2.pdf",
]

urls = [
    "https://www.feynmanlectures.caltech.edu/I_44.html"
]



vector_store = None
retriever = None
if os.path.exists(f"C:/Users/rebel/Documents/Python/python-llm-project/{vector_store_name}.parquet"):
    vector_store, retriever = vector_store_lang_chain_impl.load_vector_store()

if vector_store is None:
    vector_store, retriever = vector_store_lang_chain_impl.create_vector_store(pdf_paths, urls)

flow_controller = Controlflow(local_llm, retriever)

# question = "What is the Gibbs free energy?"
# response = flow_controller.ask_question(flow_controller.graph, question)
# print(response)

# Streamlit app
st.title("LLM Project")

question = st.text_input("Enter your question:")
if st.button("Ask"):
    response = flow_controller.ask_question(flow_controller.graph, question)
    print(response)
    st.write(response)