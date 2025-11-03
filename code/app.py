from flask import Flask, request, jsonify
import os
from chat_flow import trigger_flow

os.getcwd()+"/api_key.txt"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "g")
GROQ_MODEL = "mixtral-8x7b-8k" # Fast and powerful model on Groq
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" # The requested model for embeddings

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def handle_query():
    """Handles incoming queries from the frontend."""
    data = request.get_json()
    print(data)
    user_query = data.get('query')
    trigger_flow(user_query)
    return jsonify({"response": user_query}),200

if __name__ == '__main__':
    # This is for running locally during development
    # In a production environment, you would use a production-ready WSGI server
    app.run(debug=True)