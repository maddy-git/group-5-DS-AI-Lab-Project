from flask import Flask, request, jsonify
import requests
import faiss
import numpy as np
import os
import nltk
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Ensure NLTK resources are available for sentence tokenization
def download_nltk():
    """Downloads necessary NLTK data if not already present."""
    # We use a try-except block to handle cases where the data might not be found
    try:
        nltk.data.find('tokenizers/punkt')
    except (nltk.downloader.DownloadError, LookupError):
        print("Downloading NLTK 'punkt' resource...")
        # 'punkt' is essential for accurate sentence tokenization
        nltk.download('punkt')

download_nltk()

# ==== CONFIGURATION ====
# IMPORTANT: Replace "YOUR_GROQ_API_KEY" with your actual Groq API key.
# It is recommended to use environment variables in production.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
GROQ_MODEL = "mixtral-8x7b-8k" # Fast and powerful model on Groq
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" # The requested model for embeddings

app = Flask(__name__)
# Initialize the requested embedding model
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --- PDF Processing Function ---
def load_sentences_from_pdf(pdf_path):
    """
    Extracts text from a PDF, splits it into sentences using NLTK, and returns a list of sentences.
    
    NOTE: For this to work, you must place your PDF file (e.g., 'knowledge.pdf') 
    in the same directory as this script.
    """
    if not os.path.exists(pdf_path):
        print(f"Warning: PDF file not found at {pdf_path}. Using default documents.")
        return []

    print(f"Processing PDF: {pdf_path}...")
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            # Extract text from each page
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []

    # Use NLTK's sentence tokenizer for accurate splitting
    sentences = nltk.sent_tokenize(text)
    # Filter out empty or very short sentences
    return [s.strip() for s in sentences if len(s.strip()) > 20]

# ==== RAG KNOWLEDGE BASE SETUP ====
# We attempt to load documents from a file named 'knowledge.pdf' first.

PDF_FILE_PATH = "knowledge.pdf"
documents = load_sentences_from_pdf(PDF_FILE_PATH)

# Fallback: If no PDF is found or loading fails, use internal sample data
if not documents:
    documents = [
        "Linen pants are ideal for summer and go well with light shirts, making them perfect for hot weather.",
        "Cotton trousers are breathable and great for formal occasions, offering a classic and comfortable fit.",
        "Denim jeans pair well with casual shirts and T-shirts for a relaxed, everyday look.",
        "Wool pants are suited for cold weather and high-end formal events due to their warmth and luxurious texture.",
        "The BAAI/bge-base-en-v1.5 model is a highly performant embedding model for English text, outperforming many previous models.",
        "Groq provides incredibly fast inference using its custom Language Processing Unit (LPU) architecture, designed for speed."
    ]

print(f"Loaded {len(documents)} documents into the knowledge base.")

# Generate embeddings and set up FAISS index
doc_embeddings = embed_model.encode(documents, convert_to_numpy=True)
# The dimensionality of the BGE-base model is 768
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)
print("FAISS index created and knowledge base indexed.")


def retrieve_context(query, k=3):
    """
    Performs vector search to find the top-k most relevant documents.
    """
    # Note: query instruction for BGE models is often beneficial, but 
    # SentenceTransformer handles this automatically if trained with it.
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    
    # D: Distances, I: Indices
    D, I = index.search(query_embedding, k)
    
    # Retrieve the document chunks based on the indices
    context = [documents[i] for i in I[0] if i < len(documents)]
    return "\n\n".join(context)

def call_groq_api(prompt):
    """
    Calls the Groq API to generate a response.
    """
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY":
        return "Error: GROQ_API_KEY is not configured. Please get a key from console.groq.com."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": GROQ_MODEL,
        "temperature": 0.5
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error calling Groq API. Please check your API key, network connection, or Groq usage limits. Details: {e}"
    except KeyError:
        return "Error: Unexpected response structure from Groq API."


@app.route('/rag_query', methods=['POST'])
def rag_query():
    """
    Endpoint for running the RAG process.
    """
    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    # 1. Retrieval
    context = retrieve_context(user_query)

    # 2. Augmentation & Generation
    system_prompt = (
        "You are an intelligent assistant. Use the provided context to answer the user's question. "
        "If the answer is not found in the context, state that clearly."
    )
    
    full_prompt = (
        f"Context:\n---\n{context}\n---\n\n"
        f"Question: {user_query}"
    )

    llm_response = call_groq_api(full_prompt)

    return jsonify({
        "query": user_query,
        "context_used": context.split('\n\n'),
        "response": llm_response,
        "embedding_model": EMBEDDING_MODEL_NAME
    })

@app.route('/', methods=['GET'])
def index():
    return f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Groq RAG API Status</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {{ font-family: 'Inter', sans-serif; }}
        </style>
    </head>
    <body class="bg-gray-100 p-8">
        <div class="max-w-3xl mx-auto bg-white p-8 rounded-xl shadow-2xl">
            <h1 class="text-3xl font-bold text-gray-800 mb-6 border-b pb-2">
                Groq RAG API Status
            </h1>
            <p class="text-lg text-gray-600 mb-4">
                The RAG service is running. This application uses Groq for fast LLM inference, 
                and the **{EMBEDDING_MODEL_NAME}** model for high-quality semantic search.
            </p>

            <div class="space-y-4">
                <div class="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <h2 class="font-semibold text-xl text-blue-700">LLM Configuration</h2>
                    <p class="text-gray-700">
                        Model: <code class="font-mono bg-blue-100 p-1 rounded text-blue-800">{GROQ_MODEL}</code>
                    </p>
                    <p class="text-gray-700">
                        Status: <span class="font-medium text-green-600">Ready</span> (Check API Key placeholder)
                    </p>
                </div>
                
                <div class="p-4 bg-green-50 border border-green-200 rounded-lg">
                    <h2 class="font-semibold text-xl text-green-700">Embedding & Knowledge Source</h2>
                    <p class="text-gray-700">
                        Model: <code class="font-mono bg-green-100 p-1 rounded text-green-800">{EMBEDDING_MODEL_NAME}</code>
                    </p>
                    <p class="text-gray-700">
                        Documents: <span class="font-medium text-purple-600">{len(documents)}</span> chunks loaded.
                    </p>
                    <p class="text-sm text-gray-500 mt-2">
                        Knowledge is sourced from: **<code>knowledge.pdf</code>** (if present) or internal sample data.
                    </p>
                </div>

                <div class="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <h2 class="font-semibold text-xl text-yellow-700">API Endpoint</h2>
                    <p class="text-gray-700">
                        Endpoint: <code class="font-mono bg-yellow-100 p-1 rounded text-yellow-800">POST /rag_query</code>
                    </p>
                    <p class="text-gray-700">
                        Payload Example: <code class="font-mono bg-yellow-100 p-1 rounded text-yellow-800">{"query": "what pants are best for cold weather?"}</code>
                    </p>
                </div>
            </div>
            
            <footer class="mt-8 pt-4 border-t text-center text-sm text-gray-500">
                Built for RAG using Flask, Groq, FAISS, Sentence Transformers, and pypdf.
            </footer>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    # Flask will run the app
    # Set host to '0.0.0.0' for deployment flexibility
    app.run(host='0.0.0.0', debug=True)
