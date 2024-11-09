import unittest
from mongo_connection import MongoDBConnection

class TestMongoDBConnection(unittest.TestCase):

    def test_mongo_connection(self):
        # Create an instance of MongoDBConnection
        mongo_conn = MongoDBConnection()
        
        # Get the database and collection
        db = mongo_conn.get_database('test_database')
        collection = mongo_conn.get_collection('test_database', 'test_collection')
        
        # Check if the database and collection are not None
        self.assertIsNotNone(db)
        self.assertIsNotNone(collection)
        
        # Insert a test document
        test_data = {'test_key': 'test_value'}
        result = collection.insert_one(test_data)
        
        # Check if the document was inserted
        self.assertIsNotNone(result.inserted_id)
        
        # Retrieve the document
        retrieved_data = collection.find_one({'test_key': 'test_value'})
        
        # Check if the retrieved document matches the inserted document
        self.assertIsNotNone(retrieved_data)
        self.assertEqual(retrieved_data['test_key'], 'test_value')
        
        # Clean up the test data
        collection.delete_one({'test_key': 'test_value'})

    def test_singleton_pattern(self):
        # Create two instances of MongoDBConnection
        mongo_conn1 = MongoDBConnection()
        mongo_conn2 = MongoDBConnection()
        
        # Check if both instances are the same
        self.assertIs(mongo_conn1, mongo_conn2)

if __name__ == '__main__':
    unittest.main()