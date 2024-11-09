import ollama
from mongo_connection import MongoDBConnection
from pdf_interpreter import PDFInterpreter
from sentence_transformers import SentenceTransformer

# Pull the model
ollama.pull('llama3.1')

# MongoDB connection setup using Singleton pattern
mongo_conn = MongoDBConnection()
db = mongo_conn.get_database('rag_database')
collection = mongo_conn.get_collection('rag_database', 'rag_collection')

def chat_response(role, content):
    response = ollama.chat(model='llama3.1', messages=[
        {
            'role': role,
            'content': content,
        },
    ])
    return response

def store_pdf_text_in_mongodb(pdf_path):
    # Extract text from the PDF
    pdf_interpreter = PDFInterpreter(pdf_path)
    text = pdf_interpreter.extract_text()

    # Generate embeddings for the text
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text)

    # Prepare the data to be stored
    data = {
        'pdf_path': pdf_path,
        'text': text,
        'embeddings': embeddings.tolist()  # Convert numpy array to list for MongoDB storage
    }

    # Insert the data into MongoDB
    collection.insert_one(data)

def retrieve_pdf_text_from_mongodb(pdf_path):

    # Retrieve the data from MongoDB
    document = collection.find_one({'pdf_path': pdf_path})
    if document:
        return document['text'], document['embeddings']
    else:
        print(f"No document found for PDF path: {pdf_path}")
        return None, None
    
if __name__ == "__main__":
    pdf_path = r"C:\Users\rebel\Documents\textbooks\61-the_principles_of_chemical_equil.pdf"
    
    # Store PDF text in MongoDB
    store_pdf_text_in_mongodb(pdf_path)
    
    # Retrieve the PDF text and embeddings from MongoDB
    pdf_text, pdf_embeddings = retrieve_pdf_text_from_mongodb(pdf_path)
    print(f"Retrieved PDF text: {pdf_text[:500]}...")  # Print the first 500 characters of the retrieved text
    
    if pdf_text and pdf_embeddings:
        # Use the retrieved text and embeddings as context for the LLM
        context = f"The following text is extracted from a PDF document:\n\n{pdf_text}\n\n"
        instruction = "Summaraize the text provided."
        
        response = ollama.chat(model='llama3.1', messages=[
            {
                'role': 'system',
                'content': context,
            },
            {
                'role': 'user',
                'content': instruction,
            },
        ])
        print("LLM Response:", response['message']['content'])
    else:
        print("Failed to retrieve PDF text and embeddings from MongoDB.")