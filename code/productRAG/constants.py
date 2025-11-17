from .prompts import refinePrompts, rerankerPrompts, otherPrompts
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from code.constants import (
    CLOTH_PARQUET_PATH, CUSTOMERS_CSV_PATH, TRANSACTIONS_CSV_PATH,
    PRODUCT_EMBEDDING_PATH, GCS_BASE # Import the GCS_BASE if needed for general paths
)

# ----------------------------------------------------
# 1. GCS PATHS (Use the variables defined in constants.py)
# ----------------------------------------------------

# These should now point to your GCS paths
product_dir = CLOTH_PARQUET_PATH             # Assuming this refers to the source data
customers_dir = CUSTOMERS_CSV_PATH
transactions_dir = TRANSACTIONS_CSV_PATH
product_embedding_dir = PRODUCT_EMBEDDING_PATH

# Update the images directory path to use GCS format
# NOTE: Ensure the images are stored at this GCS location.
images_dir = f"{GCS_BASE}/images/h-and-m-personalized-fashion-recommendations/images"

# Update FAISS index paths to use GCS
# NOTE: The FAISS index files must also be stored in your GCS bucket.
product_faiss_index_dir = f"{GCS_BASE}/product_index_on_text_only.faiss"
product_faiss_dir = f"{GCS_BASE}/products_with_meta.parquet"

structured_response_prompt = otherPrompts.structured_response_prompt

# ----------------------------------------------------
# 2. Hyperparameters
# ----------------------------------------------------
temp = 0.5
top_k = 12
refine_prompt = refinePrompts.refine_prompt_balanced
reranker_prompt = rerankerPrompts.rerank_prompt_balanced
refine_prompt_name = "refine_prompt_balanced"
reranker_prompt_name = "rerank_prompt_balanced"

# ----------------------------------------------------
# 3. LLM Configuration (Secure)
# ----------------------------------------------------

# IMPORTANT SECURITY CHANGE: 
# 1. Delete the hardcoded API key assignment.
# os.environ["GOOGLE_API_KEY"] = "AIzaSyBfCV-TobP0xCdhrLvcZwbjSrIWjo03CX8" 

# 2. Instead, you MUST use a secure method:
#    a) If you are using the Gemini API key (AIza...), set it securely in the 
#       Cloud Run UI under the "Variables" tab (Recommended for simplicity).
#    b) If you are using the Cloud Run Service Account (recommended for security),
#       you will need to use a different client library (e.g., Google GenAI SDK for Vertex AI) 
#       or ensure the Service Account is authorized for the Gemini API.

# We assume the API key is set externally in the Cloud Run environment variables.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temp)
