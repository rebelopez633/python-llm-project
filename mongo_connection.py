from pymongo import MongoClient

class MongoDBConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBConnection, cls).__new__(cls)
            cls._instance.client = MongoClient('mongodb://localhost:27017/')
        return cls._instance

    def get_database(self, db_name):
        return self.client[db_name]

    def get_collection(self, db_name, collection_name):
        db = self.get_database(db_name)
        return db[collection_name]