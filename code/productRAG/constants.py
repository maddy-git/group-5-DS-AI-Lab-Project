from .prompts import refinePrompts, rerankerPrompts, otherPrompts
import os
from langchain_google_genai import ChatGoogleGenerativeAI
# ----------------------------------------------------
# GCS Configuration
# ----------------------------------------------------
# The base bucket name
BUCKET_NAME = "group-5-ai-data"

# The common prefix for all data files
GCS_DATA_BASE = f"gs://{BUCKET_NAME}/data"

# Define the full GCS paths for all data files
CLOTH_CSV_PATH = f"{GCS_DATA_BASE}/cloth_test_set.csv"
CLOTH_PARQUET_PATH = f"{GCS_DATA_BASE}/cloth_test_set.parquet"
CUSTOMERS_CSV_PATH = f"{GCS_DATA_BASE}/customers_first_3000.csv"
TRANSACTIONS_CSV_PATH = f"{GCS_DATA_BASE}/transactions_first_3000_customers.csv"

# Assuming the FAISS index and product embedding files are also in the data folder
PRODUCT_EMBEDDING_PATH = f"{GCS_DATA_BASE}/cloth_products_bge_embedded.parquet"
PRODUCT_FAISS_INDEX_PATH = f"{GCS_DATA_BASE}/product_index_on_text_only.faiss"

# Path for the images folder (used to construct public URLs in chat_flow.py)
# NOTE: The public access URL uses 'https://storage.googleapis.com/'
GCS_IMAGES_FOLDER = f"{BUCKET_NAME}/data/images/h-and-m-personalized-fashion-recommendations/images"
PUBLIC_IMAGES_BASE_URL = f"https://storage.googleapis.com/{GCS_IMAGES_FOLDER}"


# ----------------------------------------------------
# DataFrame Dictionary Keys for Consistent Access
# ----------------------------------------------------
# These keys should be used to access dataframes loaded into the dictionary in app.py
DF_CLOTH_PARQUET_KEY = 'df_cloth_parquet' 
DF_CLOTH_CSV_KEY = 'df_cloth_csv'
DF_CUSTOMERS_KEY = 'df_customers'
DF_TRANSACTIONS_KEY = 'df_transactions'



structured_response_prompt = otherPrompts.structured_response_prompt

# Hyperparameters - Product RAG
temp = 0.8
top_k = 20
refine_prompt = refinePrompts.refine_prompt_creative
reranker_prompt = rerankerPrompts.rerank_prompt_creative
refine_prompt_name = "refine_prompt_creative"
reranker_prompt_name = "rerank_prompt_creative"

#os.environ["GOOGLE_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temp)
