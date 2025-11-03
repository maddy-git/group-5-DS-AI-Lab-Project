# product_search_pipeline.py

import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from gorq_query import client  # Import Groq client directly from gorq_query.py

# --------------------------
# Paths
# --------------------------
text_embeddings_path = os.path.join(
    os.path.dirname(__file__), 
    "Product_Embedding_and_faiss_index_products_meta", 
    "product_text_embeddings_only.npy"
)
product_meta_path = os.path.join(
    os.path.dirname(__file__), 
    "Product_Embedding_and_faiss_index_products_meta", 
    "products_with_meta.parquet"
)
index_path = os.path.join(
    os.path.dirname(__file__), 
    "Product_Embedding_and_faiss_index_products_meta", 
    "product_index_on_text_only.faiss"
)
image_folder = r"../images"

# --------------------------
# Load data and embeddings
# --------------------------
products_df = pd.read_parquet(product_meta_path)

# Load text embeddings
text_embs = np.load(text_embeddings_path).astype(np.float32)
text_embs = normalize(text_embs, axis=1)  # Normalize embeddings

# Load FAISS index
loaded_index = faiss.read_index(index_path)
print("FAISS index loaded. Total vectors:", loaded_index.ntotal)

# --------------------------
# Load sentence transformer model
# --------------------------
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------
# Search products safely
# --------------------------
def search_products(query, k=5):
    """
    Encode the query, search FAISS index, and return top-k products with similarity.
    """
    # Encode query
    query_emb = text_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True  # ensure cosine similarity works
    ).astype(np.float32)

    # Search FAISS index
    distances, indices = loaded_index.search(query_emb, k)

    # Handle case where no results are returned
    if indices.shape[1] == 0:
        return pd.DataFrame(columns=list(products_df.columns) + ["similarity"])

    # Fetch results and attach similarity scores
    results = products_df.iloc[indices[0]].copy()
    results["similarity"] = distances[0]
    return results

# --------------------------
# AI judgment using Groq
# --------------------------
def ai_judge_results(user_query, user_context, retrieved_products):
    """
    Sends the query + retrieved products to Groq AI for judgment.
    Returns a dict with 'judgment' and optional 'follow_up'.
    """
    result_summary = retrieved_products[['prod_name', 'Price_INR', 'product_group_name']].to_dict(orient='records')

    prompt = f"""
You are a fashion assistant.

User query: "{user_query}"
User context: {user_context}
Retrieved products: {result_summary}

Please respond with ONE of:
- GOOD_MATCH → if the retrieved products match the user's intent
- BAD_MATCH → if they do NOT match
- If BAD_MATCH, also suggest a concise follow-up question to clarify user intent.
"""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )

    content = response.choices[0].message.content.strip()

    # Robust parsing
    if "GOOD_MATCH" in content:
        return {"judgment": "GOOD_MATCH"}
    elif "BAD_MATCH" in content:
        # Extract everything after BAD_MATCH as follow-up, or default message
        parts = content.split("BAD_MATCH", 1)
        follow_up = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "Please clarify your query."
        return {"judgment": "BAD_MATCH", "follow_up": follow_up}
    else:
        # fallback
        return {"judgment": "BAD_MATCH", "follow_up": "Please clarify your query."}

# --------------------------
# Full pipeline
# --------------------------
def handle_query(user_query, user_context, top_k=5):
    """
    Full product search pipeline:
    1. Search products using text embeddings and FAISS
    2. Send results + query context to Groq AI for judgment
    3. Return results and judgment, with optional follow-up
    """
    # Step 1: Search products
    results = search_products(user_query, k=top_k)

    # Step 2: Ask AI to judge results
    judgment = ai_judge_results(user_query, user_context, results)

    # Step 3: Return results or empty if BAD_MATCH
    if judgment["judgment"] == "GOOD_MATCH":
        return {"results": results.to_dict(orient="records"), "judgment": judgment}
    else:
        return {"results": [], "judgment": judgment}

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    user_context = [
        "Gender: M",
        "Age: 23",
        "Location: 603202",
        "--- PREVIOUS TRANSACTIONS ---",
        "Transaction 0: shirt",
        "Transaction 1: shorts"
    ]
    query = "white t-shirt for summer"

    output = handle_query(query, user_context)

    print("Pipeline output:")
    print(output)

    # If BAD_MATCH, show follow-up
    if output["judgment"]["judgment"] == "BAD_MATCH":
        print("Suggested follow-up:", output["judgment"].get("follow_up", ""))
