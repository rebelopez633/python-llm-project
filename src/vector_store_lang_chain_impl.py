import logging
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from vector_store_interface import VectorStoreInterface

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStoreLangChainImpl(VectorStoreInterface):
    @staticmethod
    def store_vector(doc_splits):
        logging.info(f"Storing document splits into vector store. Number of document splits: {len(doc_splits)}")
        
        vector_store = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
        )
        
        logging.debug(f"Stored document splits into vector store: {vector_store}")
        return vector_store
