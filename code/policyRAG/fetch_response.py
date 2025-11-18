import chromadb
from .genrate_keywords import generate_keywords_from_chunk
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "policy_docs"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(EMBED_MODEL)

def retrieve_documents(user_query, n_results=3):
    # Initialize ChromaDB client with persistence
    client = chromadb.PersistentClient(path="/home/wizard/projects/group-5-DS-AI-Lab-Project/code/chroma_db")
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # Generate keywords for the user query (this will call the LLM)
    query_keywords_list = generate_keywords_from_chunk(user_query)

    # Augment the user query with its keywords for better embedding
    augmented_query = user_query
    if query_keywords_list:
        augmented_query = ", ".join(query_keywords_list) + ". " + user_query

    # Embed the augmented user query
    query_embedding = model.encode(augmented_query, convert_to_tensor=True).cpu().numpy()

    # Query the ChromaDB collection
    results = collection.query(
        query_embeddings=[query_embedding], # ChromaDB expects a list of embeddings
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )

    # Process and return the results
    retrieved_docs = []
    if results and results['documents']:
        for i in range(len(results['documents'][0])):
            retrieved_docs.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
    return retrieved_docs

import textwrap

def format_rag_context_for_llm(query: str, max_chars: int = 800) -> str:
    """
    Formats retrieved RAG results into a structured context string for LLM input.
    
    Args:
        query (str): The user’s question.
        retrieved_docs (list): Retrieval results with 'document' and 'metadata' keys.
        max_chars (int): Max characters to include from each chunk (truncate if needed).

    Returns:
        str: Clean, formatted text suitable for LLM context.
    """
    retrieved_docs=retrieve_documents(query)
    formatted_contexts = []

    for doc in retrieved_docs:
        metadata = doc.get("metadata", {})
        policy_name = metadata.get("policy_name", "Unknown Policy")
        keywords = metadata.get("keywords", "N/A")
        content = doc.get("document", "").replace("\n", " ").strip()

        # truncate long chunks for token safety
        if len(content) > max_chars:
            content = content[:max_chars].rsplit('.', 1)[0] + "..."

        formatted = textwrap.dedent(f"""
        ─────────────────────────────
        Policy: {policy_name}
        Keywords: {keywords}
        Excerpt:
        {content}
        """).strip()

        formatted_contexts.append(formatted)

    # join all formatted blocks
    combined_context = "\n\n".join(formatted_contexts)

    # final structured prompt
    final_prompt = textwrap.dedent(f"""
    User Query: {query}

    Relevant Policy Contexts:
    {combined_context}

    Use only the information from the above context to answer clearly and factually.
    If the answer is not explicitly mentioned, respond that it is not stated in the policy.
    """).strip()

    return final_prompt


print("Retrieval function 'retrieve_documents' updated to use keywords.")

print(format_rag_context_for_llm("what is the refund policy?"))