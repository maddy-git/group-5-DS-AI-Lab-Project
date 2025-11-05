import chromadb

# Define the path to the persistent Chroma DB storage
db_path = "code/chroma_db_storage/"

try:
    # Instantiate a PersistentClient
    client = chromadb.PersistentClient(path=db_path)

    # List all collections
    collections = client.list_collections()

    print(f"Collections in '{db_path}':")
    if collections:
        for collection in collections:
            print(f"- {collection.name}")
    else:
        print("No collections found.")

except Exception as e:
    print(f"Error accessing Chroma DB or listing collections: {e}")