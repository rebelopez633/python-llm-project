import fitz  # PyMuPDF
from pymongo import MongoClient

class PDFInterpreter:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        # Open the PDF file
        document = fitz.open(self.pdf_path)
        text = ""
        # Iterate through each page
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            page_text = page.get_text()
            # print(f"Page {page_num + 1} text: {page_text[:100]}...")  # Print first 100 characters of each page
            text += page_text
        print(f"Total extracted text length: {len(text)}")
        return text
