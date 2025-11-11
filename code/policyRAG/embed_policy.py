import chromadb
from sentence_transformers import SentenceTransformer
import os
import fitz

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
    ids = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        ids.append(str(uuid.uuid4()))
        start += chunk_size - overlap
    return chunks, ids

if pdf_text:
    simple_text_chunks, simple_chunk_ids = simple_chunk_text_with_ids(pdf_text, chunk_size=700, overlap=70)
    print(f"Created {len(simple_text_chunks)} text chunks.")
else:
    simple_text_chunks = []
    simple_chunk_ids = []

# --- 3. Load Embedding Model (from cell 4cc0f633) ---
# Assuming model is available from previous steps, otherwise load it
try:
    model
except NameError:
    model_name = "BAAI/bge-base-en-v1.5"
    model = SentenceTransformer(model_name)
    print(f"Loaded embedding model: {model_name}")

# --- 4. Setup Persistent Chroma DB and Collections ---
persistent_storage_path = os.getcwd() +  "/chroma_db_storage"
client = chromadb.PersistentClient(path=persistent_storage_path)

# Define collection names for different strategies
collection_name_default = 'faq_and_policy_default_chunking'
collection_name_hybrid = 'faq_and_policy_hybrid_search' # Using the name from the successful hybrid search experiment


# Get or create collections
collection_default = client.get_or_create_collection(name=collection_name_default)
collection_hybrid = client.get_or_create_collection(name=collection_name_hybrid)

print(f"Collection '{collection_default.name}' loaded or created.")
print(f"Collection '{collection_hybrid.name}' loaded or created.")

# --- 5. Populate Collections (if empty) ---
if simple_text_chunks:
    # Populate default collection
    if collection_default.count() == 0:
        print(f"Collection '{collection_default.name}' is empty. Populating...")
        chunk_embeddings_default = model.encode(simple_text_chunks, convert_to_tensor=False)
        collection_default.add(
            documents=simple_text_chunks,
            embeddings=chunk_embeddings_default.tolist(),
            ids=simple_chunk_ids
        )
        print(f"Added {len(simple_text_chunks)} chunks to '{collection_default.name}'.")
    else:
        print(f"Collection '{collection_default.name}' already contains {collection_default.count()} items.")


    # Populate hybrid collection (using the same chunks and IDs for consistency with the fixed hybrid query function)
    if collection_hybrid.count() == 0:
        print(f"Collection '{collection_hybrid.name}' is empty. Populating...")
        chunk_embeddings_hybrid = model.encode(simple_text_chunks, convert_to_tensor=False) # Using the same model for hybrid as well
        collection_hybrid.add(
            documents=simple_text_chunks,
            embeddings=chunk_embeddings_hybrid.tolist(),
            ids=simple_chunk_ids
        )
        print(f"Added {len(simple_text_chunks)} chunks to '{collection_hybrid.name}'.")
    else:
         print(f"Collection '{collection_hybrid.name}' already contains {collection_hybrid.count()} items.")

else:
    print("No chunks available to populate collections.")