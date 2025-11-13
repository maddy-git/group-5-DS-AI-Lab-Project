from . import constants

import pandas as pd
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer, util

cloth_data = pd.read_parquet(constants.product_embedding_dir)
model = SentenceTransformer("BAAI/bge-small-en")

model_faiss = SentenceTransformer("all-MiniLM-L6-v2")
# faiss_cloth_data = pd.read_parquet(constants.product_faiss_dir)
# faiss_cloth_index_data = faiss.read_index(constants.product_faiss_index_dir)


refine_chain = constants.refine_prompt | constants.llm


def search_products(customer_query, customer_context, chat_history, top_k = 20):
    refined = refine_chain.invoke({
        "customer_context": customer_context,
        "chat_history": chat_history,
        "query": customer_query
    })
    refined_query = refined.content.strip()
    print(f"\n🎯 Refined Query (Personalized):\n{refined_query}\n")

    query_embedding = model.encode(refined_query, normalize_embeddings=True)
    product_embeddings = np.vstack(cloth_data["embeddings"].to_numpy()).astype(np.float32)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    query_tensor = torch.tensor(query_embedding).unsqueeze(0)
    product_tensor = torch.tensor(product_embeddings)
    scores = util.cos_sim(query_tensor, product_tensor)[0].cpu().numpy()

    top_k_idx = np.argsort(scores)[::-1][:top_k]
    top_matches = cloth_data.iloc[top_k_idx].copy()
    top_matches["similarity_score"] = scores[top_k_idx]

    print(f"✅ Found {top_k} best-matching products for customer.\n")

    matches =  top_matches[[
        "prod_name", "product_type_name", "product_group_name",
        "colour_group_name", "department_name", "Price_INR",
        "Discount_Percentage", "Return_Type", "SKU_No"
    ]]

    print(matches)
    return matches

# def search_products_faiss(customer_query, customer_context, chat_history, top_k = 20, query_emb=None):
#     refined = refine_chain.invoke({
#         "customer_context": customer_context,
#         "chat_history": chat_history,
#         "query": customer_query
#     })
#     refined_query = refined.content.strip()
#     print(f"\n🎯 Refined Query (Personalized):\n{refined_query}\n")
#
#     query_emb = model_faiss.encode([refined_query], normalize_embeddings=True).astype(np.float32)[0]
#     distances, indices = faiss_cloth_index_data.search(np.expand_dims(query_emb.astype(np.float32), axis=0), top_k)
#     valid_idxs = indices[0][indices[0] >= 0]
#     results = faiss_cloth_data.iloc[valid_idxs].copy().reset_index(drop=True)
#     results["similarity"] = distances[0][: len(valid_idxs)]
#
#     print(f"✅ Found {top_k} best-matching products for customer.\n")
#
#     matches = results[[
#         "prod_name", "product_type_name", "product_group_name",
#         "colour_group_name", "department_name", "Price_INR",
#         "Discount_Percentage", "Return_Type", "SKU_No"
#     ]]
#
#     print(matches)
#     return matches