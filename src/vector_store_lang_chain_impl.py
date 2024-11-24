import logging
from load_document_lang_chain_impl import LoadDocumentLangChainImpl
from text_splitter_lang_chain_impl import TextSplitterLangChainImpl
from langchain_community.vectorstores import SKLearnVectorStore

from vector_store_interface import VectorStoreInterface


class VectorStoreLangChainImpl(VectorStoreInterface):
    def __init__(self, embedding, vector_store_name, persist_location="C:/Users/rebel/Documents/Python/python-llm-project"):
        self.embedding = embedding
        self.vector_store_name = vector_store_name
        self.persist_location = persist_location
    
    def create_vector_store(self, pdf_paths, urls):
        
        docs_list = LoadDocumentLangChainImpl.load_web(urls)
        docs_list += LoadDocumentLangChainImpl.load_pdf(pdf_paths)
        doc_splits = TextSplitterLangChainImpl.split_text(docs_list)

        logging.info(f"Storing document splits into vector store. Number of document splits: {len(doc_splits)}")
        vector_store = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=self.embedding,
            persist_path=f"{self.persist_location}/{self.vector_store_name}.parquet",
            serializer="parquet"
        )
        
        logging.info(f"Persisting vector store to {self.persist_location}/{self.vector_store_name}.parquet")
        vector_store.persist()

        logging.info(f"Creating retriever from vector store")
        retriever = vector_store.as_retriever()

        return vector_store, retriever
    
    def load_vector_store(self):
        
        # Load the vector store from disk
        vectorstore = SKLearnVectorStore(
            embedding=self.embedding,
            persist_path=f"{self.persist_location}/{self.vector_store_name}.parquet",
            serializer="parquet"
        )

        # create retriever
        retriever = vectorstore.as_retriever()

        return vectorstore, retriever