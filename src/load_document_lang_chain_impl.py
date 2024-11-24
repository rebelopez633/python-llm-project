import logging
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from load_document_interface import LoadDocumentInterface
from requests.exceptions import SSLError

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
        docs = []
        for url in urls:
            logging.info(f"Loading web document from: {url}")
            try:
                docs.extend(WebBaseLoader(url).load())
            except SSLError as ssl_error:
                logging.error(f"SSL error: {ssl_error}\nAttempting without SSL verification")
                docs.append(WebBaseLoader(url, verify_ssl=False).load())
            flattened_docs = [item for sublist in docs for item in sublist]
        logging.debug(f"Loaded web documents: {flattened_docs}")
        return flattened_docs
