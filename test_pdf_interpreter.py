import unittest
from pdf_interpreter import PDFInterpreter

class TestPDFInterpreter(unittest.TestCase):

    def test_extract_text(self):
        # Path to a sample PDF file for testing
        pdf_path = r"C:\Users\rebel\Documents\textbooks\repertorium_gibbs2.pdf"
        
        # Create an instance of PDFInterpreter
        pdf_interpreter = PDFInterpreter(pdf_path)
        
        # Extract text from the PDF
        text = pdf_interpreter.extract_text()
        
        # Check if the extracted text is not empty
        self.assertIsNotNone(text)
        self.assertTrue(len(text) > 0)

if __name__ == '__main__':
    unittest.main()