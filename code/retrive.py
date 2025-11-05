import chromadb
from sentence_transformers import SentenceTransformer
import os

def simple_retrieve_from_chroma(query_text, collection_name='faq_and_policy_hybrid_search', db_path="code/chroma_db_storage", n_results=5):
    """
    Retrieves relevant documents from a Chroma DB collection based on a user query.

    Args:
        query_text: The user's query string.
        collection_name: The name of the Chroma DB collection.
        db_path: The path to the persistent Chroma DB storage.
        n_results: The number of results to retrieve.

    Returns:
        A list of retrieved document strings.
    """
    try:
        # Instantiate a PersistentClient
        client = chromadb.PersistentClient(path=db_path)

        # Get the collection
        collection = client.get_collection(name=collection_name)
        print(f"Successfully retrieved collection: '{collection.name}' from '{db_path}'")

        # Load the embedding model (assuming it's available or load it if not)
        try:
            model # Check if model is already defined
        except NameError:
            model_name = "BAAI/bge-base-en-v1.5"
            model = SentenceTransformer(model_name)
            print(f"Loaded embedding model: {model_name}")


        # Generate embedding for the query
        query_embedding = model.encode(query_text, convert_to_tensor=False).tolist()

        # Query the collection
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents']
        )

        # Extract and return the document strings
        if results and 'documents' in results and results['documents']:
            return results['documents'][0]
        else:
            return []

    except Exception as e:
        print(f"Error during retrieval: {e}")
        return [f"Error during retrieval: {e}"]

import re

# Test the simple retrieval function and compile results into a paragraph
sample_query = "what are the payment methods available"
retrieved_documents = simple_retrieve_from_chroma(sample_query)

print("\nRetrieved Documents:")
if retrieved_documents and not isinstance(retrieved_documents[0], str) and "Error" in retrieved_documents[0]:
    print(retrieved_documents[0]) # Print the error message
    compiled_context = "" # Set context to empty if there was an error
else:
    # Compile retrieved documents into a single paragraph with a character limit and clean up spacing
    compiled_context_list = []
    character_limit = 3000
    current_length = 0

    for doc in retrieved_documents:
        # Clean up the document text: replace multiple newlines/spaces with single space, strip leading/trailing spaces
        cleaned_doc = ' '.join(doc.split()).strip()

        # Remove emojis and bullet points
        # This regex removes common bullet points (like -, *, •) and Unicode emojis
        cleaned_doc = re.sub(r'[\u2022\u2023\u2043\u002D\u002A\u200B-\u200F\u2028-\u202F\u2060-\u206F\uFEFF\uFFFE\uFFFF]', '', cleaned_doc) # Remove bullet points and zero-width spaces
        cleaned_doc = re.sub(r'[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0000200D]', '', cleaned_doc) # Remove emojis


        # Check if adding this cleaned document exceeds the character limit
        # Add 1 for the space that will be added between documents
        if current_length + len(cleaned_doc) + (1 if compiled_context_list else 0) <= character_limit:
            compiled_context_list.append(cleaned_doc)
            current_length += len(cleaned_doc) + (1 if compiled_context_list else 0)
        else:
            # If adding the whole document exceeds the limit, add a partial document if possible
            remaining_characters = character_limit - current_length - (1 if compiled_context_list else 0)
            if remaining_characters > 0:
                # Add a partial cleaned document
                partial_doc = cleaned_doc[:remaining_characters]
                # Ensure we don't cut off mid-word, find the last space
                last_space = partial_doc.rfind(' ')
                if last_space != -1:
                    compiled_context_list.append(partial_doc[:last_space] + '...') # Add ellipsis
                else:
                     compiled_context_list.append(partial_doc + '...') # Add ellipsis even if mid-word
                current_length += len(compiled_context_list[-1]) + (1 if len(compiled_context_list) > 1 else 0)

            break # Stop adding documents once the limit is reached

    # Join the cleaned and potentially truncated documents into a single paragraph
    compiled_context = " ".join(compiled_context_list)

    if not compiled_context:
        compiled_context = "No relevant documents retrieved to form a context within the character limit."

print("\n--- Compiled Context (for LLM) ---")
print(compiled_context)
print("\n--- End of Compiled Context ---")

# Check the length of the compiled context
context_length = len(compiled_context)
print(f"\nCompiled context character count: {context_length}.")
print(f"Context character limit: {character_limit}.")