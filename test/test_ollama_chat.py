import unittest
from pymongo import MongoClient
from ollama_chat import chat_response, store_pdf_text_in_mongodb, retrieve_pdf_text_from_mongodb
from mongo_connection import MongoDBConnection

class TestOllamaChat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up MongoDB connection
        cls.mongo_conn = MongoDBConnection()
        cls.db = cls.mongo_conn.get_database('test_database')
        cls.collection = cls.mongo_conn.get_collection('test_database', 'test_collection')

    @classmethod
    def tearDownClass(cls):
        # Clean up the database after tests
        cls.db.drop_collection('test_collection')

    def test_chat_response(self):
        response = chat_response('user', 'Hello, how are you?')

        # Check if the response contains the 'message' key
        self.assertIn('message', response)
        # Check if the 'message' contains the 'content' key
        self.assertIn('content', response['message'])
        # Check if the content is not empty
        self.assertTrue(len(response['message']['content']) > 0)

    def test_store_pdf_text_in_mongodb(self):
        pdf_path = r"C:\Users\rebel\Documents\textbooks\repertorium_gibbs2.pdf"
        store_pdf_text_in_mongodb(pdf_path)

        # Retrieve the PDF text and embeddings from MongoDB
        text, embeddings = retrieve_pdf_text_from_mongodb(pdf_path)
        
        # Check if the text and embeddings are not None
        self.assertIsNotNone(text)
        self.assertIsNotNone(embeddings)
        self.assertTrue(len(text) > 0)
        self.assertTrue(len(embeddings) > 0)

if __name__ == '__main__':
    unittest.main()