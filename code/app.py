from flask import Flask, request, jsonify
from chat_flow import trigger_flow
import os
import gdown
import pandas as pd

app = Flask(__name__)

# ---------------------------
# Secure Path Handling
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

parquet_path = os.path.join(DATA_DIR, "cloth_test_set.parquet")

# ---------------------------
# Google Drive Download
# ---------------------------
FILE_ID = "1r8D9MMKfzHO6HounJxXvnog2Y9oxdrIj"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(parquet_path):
    print("Downloading dataset from Google Drive...")
    gdown.download(URL, parquet_path, quiet=False)
else:
    print("Dataset already present.")

# ---------------------------
# Load Parquet File
# ---------------------------
df = pd.read_parquet(parquet_path)
print("Parquet file loaded successfully.")


# ---------------------------
# API Endpoint
# ---------------------------
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()

    user_query = data.get('query')
    llm_response, images = trigger_flow(user_query)

    return jsonify({
        "response": llm_response,
        "images": images
    }), 200


# ---------------------------
# Run App (local mode)
# ---------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
