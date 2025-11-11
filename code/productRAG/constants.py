from .prompts import refinePrompts, rerankerPrompts, otherPrompts
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Directories
product_dir = '/Users/manish-17509/PycharmProjects/data/cloth_test_set.csv'
customers_dir = '/Users/manish-17509/PycharmProjects/data/customers_first_3000.csv'
transactions_dir = '/Users/manish-17509/PycharmProjects/data/transactions_first_3000_customers.csv'
product_embedding_dir = '/Users/manish-17509/PycharmProjects/data/cloth_test_set.parquet'
images_dir = '/Users/manish-17509/PycharmProjects/data/images/h-and-m-personalized-fashion-recommendations/images'

#product_test_dir = '/Users/manish-17509/PycharmProjects/data/cloth_test_set.csv'
#product_test_embedding_dir = '/Users/manish-17509/PycharmProjects/data/cloth_test_set.parquet'

product_faiss_index_dir = '/Users/manish-17509/PycharmProjects/data/product_index_on_text_only.faiss'
product_faiss_dir = '/Users/manish-17509/PycharmProjects/data/products_with_meta.parquet'

structured_response_prompt = otherPrompts.structured_response_prompt

# Hyperparameters - Product RAG
temp = 0.5
top_k = 12
refine_prompt = refinePrompts.refine_prompt_balanced
reranker_prompt = rerankerPrompts.rerank_prompt_balanced
refine_prompt_name = "refine_prompt_balanced"
reranker_prompt_name = "rerank_prompt_balanced"

os.environ["GOOGLE_API_KEY"] = ""
#os.environ["GOOGLE_API_KEY"] = ""
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temp)