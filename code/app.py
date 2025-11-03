from flask import Flask, request, jsonify
import os
from chat_flow import trigger_flow
import chromadb
from sentence_transformers import SentenceTransformer


os.getcwd()+"/api_key.txt"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "g")
GROQ_MODEL = "mixtral-8x7b-8k" # Fast and powerful model on Groq
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" # The requested model for embeddings


model = SentenceTransformer(EMBEDDING_MODEL_NAME)

app = Flask(__name__)

def retrive_policy():
    client = chromadb.PersistentClient(path=os.getcwd()+ "/chroma_db_storage")
    sample_question = "what are the payment meths available?"

    # Generate embedding for the sample question
    question_embedding = model.encode(sample_question, convert_to_tensor=False)

    collection = client.get_collection(name="faq_and_policy")

    # Query the collection to find similar documents
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=3  # Get the top 3 most similar results
    )

    # Print the results
    print("Query Results:")
    for i, doc in enumerate(results['documents'][0]):
        print(f"Result {i+1}:")
        print(f"  Document: {doc}")
        print(f"  Distance: {results['distances'][0][i]}")
        print("-" * 20)

@app.route('/query', methods=['POST'])
def handle_query():
    """Handles incoming queries from the frontend."""
    data = request.get_json()
    print(data)
    user_query = data.get('query')
    retrive_policy()
    trigger_flow(user_query)
    return jsonify({"response": user_query}),200

if __name__ == '__main__':
    # This is for running locally during development
    # In a production environment, you would use a production-ready WSGI server
    app.run(debug=True)