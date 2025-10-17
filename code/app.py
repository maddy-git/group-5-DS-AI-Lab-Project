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

PDF_FILE_PATH = "milestones/milestone 2/faq_and_policy.pdf"
documents = load_sentences_from_pdf(PDF_FILE_PATH)

# ==== Sample knowledge base for RAG ====
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


# ==== API route ====
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("query", "")

    # RAG: get similar context
    context_docs = retrieve_context(user_query, top_k=2)
    context = " ".join(context_docs)

    prompt = f"""You are a fashion stylist AI.
    Context: {context}
    Question: {user_query}
    Give a natural, detailed answer including fabric, color, and season suitability."""

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    res = requests.post("https://api.groq.com/openai/v1/chat/completions",
                        headers=headers, json=payload)
    answer = res.json()["choices"][0]["message"]["content"]
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
