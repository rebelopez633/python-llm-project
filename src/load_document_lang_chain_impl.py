import logging
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from load_document_interface import LoadDocumentInterface

class LoadDocumentLangChainImpl(LoadDocumentInterface):

    @staticmethod
    def load_pdf(paths: list[str]) -> list[any]:
        logging.info(f"Loading PDF documents from paths: {paths}")
        docs = [PyPDFLoader(path).load() for path in paths]
        flattened_docs = [item for sublist in docs for item in sublist]
        logging.debug(f"Loaded PDF documents: {flattened_docs}")
        return flattened_docs
    
    @staticmethod
    def load_web(urls: list[str]) -> list[any]:
        logging.info(f"Loading web documents from URLs: {urls}")
        docs = [WebBaseLoader(url).load() for url in urls]
        flattened_docs = [item for sublist in docs for item in sublist]
        logging.debug(f"Loaded web documents: {flattened_docs}")
        return flattened_docs
