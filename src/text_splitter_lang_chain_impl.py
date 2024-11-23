import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from text_splitter_interface import TextSplitterInterface

class TextSplitterLangChainImpl(TextSplitterInterface):

    @staticmethod
    def split_text(text: list[any]) -> list[str]:
        logging.info(f"Splitting text into chunks. Number of documents: {len(text)}")
        
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(text)
        
        logging.debug(f"Text split into {len(chunks)} chunks.")
        return chunks