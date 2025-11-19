import chromadb
from sentence_transformers import SentenceTransformer
import fitz # PyMuPDF
import re
from genrate_keywords import generate_keywords_from_chunk
import os


client = chromadb.Client()


# Load the embedding model
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(EMBED_MODEL)
PDF_PATH = os.getcwd() + "/code/content/faq_and_policy-v3.pdf"
COLLECTION_NAME = "policy_docs"




# 1️⃣ Extract text using PyMuPDF
def extract_text(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    text+="\n==========="
    return text

# 2️⃣ Split by ### headers (title line followed by ###)
def split_policies(text):
    # Regex to capture policy name immediately following '###' and its content until the next '###' or end of string.
    # re.DOTALL allows '.' to match newlines, important for content that spans multiple lines.
    pattern = r'###\s*(?P<policy_name>[^\n]+)\n(?P<content>.*?)(?=\n###|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)
    policies = []
    for i, (policy_name, content) in enumerate(matches, 1):
        policies.append({
            "policy_id": f"POL_{i:03}",
            "policy_name": policy_name.strip(),
            "content": content.strip()
        })
    return policies

# 3️⃣ Chunk each policy into ~600-word segments
def chunk_policy_content(policy, chunk_size=200):
    words = policy["content"].split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i + chunk_size])
        chunks.append({
            "policy_id": policy["policy_id"],
            "policy_name": policy["policy_name"],
            "chunk_id": f"{policy['policy_id']}_CH{i//chunk_size+1}",
            "content": chunk_text
        })
    return chunks

pdf_content = extract_text(PDF_PATH)
policies = split_policies(pdf_content)


# 5️⃣ Embed and store in ChromaDB
def store_in_chromadb(policies):
    # Initialize ChromaDB client with persistence
    db_path = os.getcwd() + "/code/chroma_db"
    client = chromadb.PersistentClient(path=db_path)

    # Clear the existing collection before adding new data to ensure freshness
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Existing collection '{COLLECTION_NAME}' deleted.")
    except Exception as e:
        print(f"Collection '{COLLECTION_NAME}' did not exist or could not be deleted. Creating a new one. Error: {e}")

    collection = client.create_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' created.")

    all_docs, all_metas, all_ids, all_embeddings = [], [], [], []

    for policy in policies:
        # Chunk each policy's content
        chunks = chunk_policy_content(policy)
        print(f"Policy: '{policy['policy_name']}' (Content Length: {len(policy['content'].split())} words) -> Generated {len(chunks)} chunks.")

        for chunk in chunks:
            # Generate keywords for the current chunk
            keywords = generate_keywords_from_chunk(chunk["content"])
            # Convert the list of keywords to a single string for ChromaDB metadata
            keywords_str = ", ".join(keywords)

            # Generate embedding using the SentenceTransformer model loaded in a previous cell
            emb = model.encode(chunk["content"], convert_to_tensor=True).cpu().numpy()

            all_docs.append(chunk["content"])
            all_ids.append(chunk["chunk_id"])
            all_metas.append({
                "policy_id": chunk["policy_id"],
                "policy_name": chunk["policy_name"],
                "keywords": keywords_str # Store keywords as a single string
            })
            all_embeddings.append(emb)

    collection.add(
        documents=all_docs,
        metadatas=all_metas,
        ids=all_ids,
        embeddings=all_embeddings
    )
    print(f"✅ Stored {len(all_docs)} chunks across {len(policies)} policies.")

# Call the function to store the policies in ChromaDB
store_in_chromadb(policies)