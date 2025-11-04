import fitz # PyMuPDF
import os 
import chromadb
from sentence_transformers import SentenceTransformer


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

pdf_path = os.getcwd() + "/content/faq_and_policy.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

print(f"Extracted {len(pdf_text)} characters from the PDF.")

client = chromadb.PersistentClient(path="./chroma_db_storage")

# Load the embedding model
model_name = "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(model_name)

# Generate embeddings
embeddings = model.encode(pdf_text, convert_to_tensor=True)

print(f"Generated embeddings with shape: {embeddings.shape}")

# Create a collection named 'faq_and_policy'
collection = client.get_or_create_collection(name='faq_and_policy')

print(f"Collection '{collection.name}' created or retrieved.")

import uuid

def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

text_chunks = chunk_text(pdf_text)
chunk_ids = [str(uuid.uuid4()) for _ in text_chunks]

chunk_embeddings = model.encode(text_chunks, convert_to_tensor=False)

collection.add(
    documents=text_chunks,
    embeddings=chunk_embeddings.tolist(), # Convert numpy array to list for ChromaDB
    ids=chunk_ids
)

print(f"Added {len(text_chunks)} chunks to the collection.")


