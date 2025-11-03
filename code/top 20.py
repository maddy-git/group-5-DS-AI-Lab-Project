import pandas as pd

def build_customer_context(cust_id: int):
    """
    Independent function — creates a complete context list for a customer.
    Includes:
      - Customer info (excluding 'size', 'cust_id', and 'embed_text')
      - All related transactions
      - All related product details
    """

    # --- Load required datasets ---
    customers = pd.read_csv("/content/customers_first_3000.csv")
    transactions = pd.read_csv("/content/transactions_first_3000_customers.csv")
    cloth_data = pd.read_csv("/content/cloth_products_for_3000_customers.csv")

    # --- Initialize context ---
    context = []

    # --- CUSTOMER INFO ---
    if cust_id not in customers["cust_id"].values:
        return [f"❌ Customer ID {cust_id} not found in dataset."]

    customer_info = customers[customers["cust_id"] == cust_id].iloc[0]

    # Exclude unnecessary columns
    excluded_cols = ["size", "cust_id", "embed_text"]

    context.append("--- CUSTOMER CONTEXT ---")
    for col, val in customer_info.items():
        if col not in excluded_cols:
            context.append(f"{col}: {val}")
    context.append("---------------------------")

    # --- TRANSACTIONS ---
    customer_transactions = transactions[transactions["cust_id"] == cust_id]
    context.append("--- PURCHASE TRANSACTIONS ---")

    if customer_transactions.empty:
        context.append("No transactions found for this customer.")
    else:
        for i, row in customer_transactions.iterrows():
            tx_details = ", ".join([
                f"{col}: {row[col]}" for col in customer_transactions.columns if col != "cust_id"
            ])
            context.append(f"Transaction {i}: {tx_details}")

    context.append("---------------------------")

    # --- PRODUCTS PURCHASED ---
    purchased_skus = customer_transactions["SKU_No"].unique()
    purchased_products = cloth_data[cloth_data["SKU_No"].isin(purchased_skus)]

    context.append("--- PRODUCTS PURCHASED ---")

    if purchased_products.empty:
        context.append("No product details found for this customer's purchases.")
    else:
        for i, row in purchased_products.iterrows():
            prod_details = ", ".join([
                f"{col}: {row[col]}" for col in purchased_products.columns
            ])
            context.append(f"Product {i}: {prod_details}")

    context.append("---------------------------")

    print(f"✅ Context successfully created for customer {cust_id}")
    return context


# --- Example Usage ---
cust_id = 5
context = build_customer_context(cust_id)

# Display context neatly
print("\n".join(context))








# ======================================================
# 🧠 Context-Aware Product Retrieval Pipeline (Top 20 Results)
# Automatically builds customer context from cust_id
# ======================================================

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from langchain_core.prompts import ChatPromptTemplate

# --- Step 1: Load embedded product dataset ---
cloth_data = pd.read_parquet("/content/cloth_products_bge_embedded.parquet")

# Load the same embedding model used for embeddings
model = SentenceTransformer("BAAI/bge-small-en")

print(f"✅ Loaded {len(cloth_data)} products with embeddings.\n")


# --- Step 2: Updated LLM Refinement Chain ---
refine_prompt = ChatPromptTemplate.from_template("""
You are a smart and detail-oriented fashion assistant.
Your job is to rewrite the user's shopping query so that it reflects
not only the customer's personal details, but also their previous purchase history.

### INSTRUCTIONS:
1. Give **highest priority** to the user's current query intent.
2. Integrate relevant information from the customer's context — including:
   - Gender, Age, Location (from customer details)
   - Purchase history (from their past transactions and product data)
3. Use their past purchases to infer style, category, or color preferences if mentioned in context.
4. Make the refined query more specific and personalized.

### EXAMPLE
Context:
--- CUSTOMER CONTEXT ---
Gender: Female
Age: 25-30
Postal Code: Mumbai
--- PURCHASE TRANSACTIONS ---
Transaction 0: Purchased formal shirts and trousers
--- PRODUCTS PURCHASED ---
Product 0: Cotton Slim Shirt, Color: Blue
Product 1: Linen Regular Fit Pant, Color: Beige

User Query:
show me something to wear to a party

Refined Query:
show me trendy but elegant party outfits for a 25-year-old female from Mumbai, 
preferably in styles similar to her past purchases like shirts and pants, in blue or beige tones.

---

### INPUTS
Context:
{context}

User Query:
{query}

Now return **only the refined query text**, nothing else.
""")

refine_chain = refine_prompt | llm


# --- Step 3: Product Search Function (Now uses cust_id instead of context) ---
def search_products(customer_query: str, cust_id: int, top_k: int = 20):
    """
    1. Builds context automatically from the given customer ID (using build_customer_context()).
    2. Refines the user query using LLM and the built context.
    3. Embeds the refined query using BGE.
    4. Computes cosine similarity with product embeddings.
    5. Returns top-k most relevant products (default = 20).
    """

    # ✅ Step 1: Build context automatically
    context = build_customer_context(cust_id)

    # ✅ Step 2: Refine query using LLM and context
    refined = refine_chain.invoke({
        "context": "\n".join(context),
        "query": customer_query
    })
    refined_query = refined.content.strip()
    print(f"\n🎯 Refined Query (Personalized):\n{refined_query}\n")

    # ✅ Step 3: Encode refined query into embedding vector
    query_embedding = model.encode(refined_query, normalize_embeddings=True)

    # ✅ Step 4: Prepare product embeddings (dtype consistency fix)
    product_embeddings = np.vstack(cloth_data["embeddings"].to_numpy()).astype(np.float32)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # ✅ Step 5: Convert to torch tensors
    query_tensor = torch.tensor(query_embedding).unsqueeze(0)
    product_tensor = torch.tensor(product_embeddings)

    # ✅ Step 6: Compute cosine similarity
    scores = util.cos_sim(query_tensor, product_tensor)[0].cpu().numpy()

    # ✅ Step 7: Sort & get top-k indices
    top_k_idx = np.argsort(scores)[::-1][:top_k]
    top_matches = cloth_data.iloc[top_k_idx].copy()
    top_matches["similarity_score"] = scores[top_k_idx]

    print(f"✅ Found {top_k} best-matching products for customer {cust_id}.\n")

    return top_matches[[
        "SKU_No", "prod_name", "product_type_name", "product_group_name",
        "colour_group_name", "department_name", "Price_INR", "Eco_Score",
        "Discount_Percentage", "Return_Type", "similarity_score"
    ]]


# --- Step 4: Example Usage ---
customer_query = "show me something for office wear"
cust_id = 5  # 👈 change to any valid customer ID

top_products = search_products(customer_query, cust_id, top_k=20)

print("\n👕 Top 20 Recommended Products:\n")
print(top_products.to_string(index=False))
