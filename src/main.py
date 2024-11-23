import os
import logging
from load_document_lang_chain_impl import LoadDocumentLangChainImpl
from text_splitter_lang_chain_impl import TextSplitterLangChainImpl
from vector_store_lang_chain_impl import VectorStoreLangChainImpl
from control_flow import Controlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["TAVILY_API_KEY"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "local-llama32-rag"

pdf_path = r"C:\Users\rebel\Documents\textbooks\61-the_principles_of_chemical_equil.pdf"
local_llm = "llama3.2:3b-instruct-fp16"

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs_list = LoadDocumentLangChainImpl.load_web(urls)
doc_splits = TextSplitterLangChainImpl.split_text(docs_list)

vector_store = VectorStoreLangChainImpl.store_vector(doc_splits)
logging.debug("Creating retriever from vector store.")
retriever = vector_store.as_retriever(k=3)
logging.debug("Retriever created from vector store.")

logging.debug("Initializing Controlflow.")
flow_controller = Controlflow(local_llm, retriever)
logging.debug("Controlflow initialized.")

inputs = {"question": "What are the types of agent memory?", "max_retries": 3}

flow_controller.stream_events(flow_controller.graph, inputs)