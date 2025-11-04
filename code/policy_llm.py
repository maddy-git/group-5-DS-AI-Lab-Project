import chromadb
import os
from sentence_transformers import SentenceTransformer

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from productRAG.constants import llm

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" # The requested model for embeddings
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

client = chromadb.PersistentClient(path=os.getcwd()+ "/chroma_db_storage")

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

def retrieve_policy(query):
    # Generate embedding for the sample question
    question_embedding = model.encode(query, convert_to_tensor=False)
    collection = client.get_collection(name="faq_and_policy")

    # Query the collection to find similar documents
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=3  # Get the top 3 most similar results
    )

    return results['documents'][0]

def answer_policy_or_return_query(query, history):
    response = rag_chain.invoke({
        "history": history,
        "context": retrieve_policy(query),
        "query": query
    })

    return response