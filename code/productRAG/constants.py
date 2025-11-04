from .prompts import refinePrompts, rerankerPrompts, otherPrompts
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Directories
product_dir = '/Users/manish-17509/PycharmProjects/data/cloth_products_for_3000_customers.csv'
customers_dir = '/Users/manish-17509/PycharmProjects/data/customers_first_3000.csv'
transactions_dir = '/Users/manish-17509/PycharmProjects/data/transactions_first_3000_customers.csv'
product_embedding_dir = '/Users/manish-17509/PycharmProjects/data/cloth_products_bge_embedded.parquet'

structured_response_prompt = otherPrompts.structured_response_prompt

# Hyperparameters - Product RAG
temp = 0.2
top_k = 5
refine_prompt = refinePrompts.refine_prompt_contextual
reranker_prompt = rerankerPrompts.rerank_prompt_contextual
refine_prompt_name = "refine_prompt_contextual"
reranker_prompt_name = "rerank_prompt_contextual"

#os.environ["GOOGLE_API_KEY"] = "AIzaSyBfCV-TobP0xCdhrLvcZwbjSrIWjo03CX8"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDsqn5BtaPWEI4HE_4gAANWVSpMlxClE8Y"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temp)