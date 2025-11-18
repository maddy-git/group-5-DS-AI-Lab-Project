import chromadb
import os
from sentence_transformers import SentenceTransformer

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from productRAG.constants import llm
from policyRAG.fetch_response import format_rag_context_for_llm

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" # The requested model for embeddings
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

client = chromadb.PersistentClient(path=os.getcwd()+ "/code/chroma_db_storage")

# --- 1. Define the Prompt Template ---
template = """
    **System Instructions:**
    You are an expert Question-Answering assistant. Your goal is to answer the final "User Query" based *only* on the provided "Relevant Context Chunks" and the "Conversation History." Do not use external knowledge.

    **Conversation History:**
    {history}

    **Relevant Context Chunks:**
    {context}

    **User Query:**
    {query}

    **Final Answer:**
    """

# --- 2. Instantiate Components ---
prompt = ChatPromptTemplate.from_template(template)

# --- 3. Build the Chain ---
rag_chain = prompt | llm | StrOutputParser()



def answer_policy_or_return_query(query, history):
    response = rag_chain.invoke({
        "history": history,
        "context": format_rag_context_for_llm(query),
        "query": query
    })

    return response