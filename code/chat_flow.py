from classify_query import classify_query
from memory_init import memory, add_chat, refresh, get_history


def trigger_flow(query):
    classification = classify_query(query, get_history())
    if classification == "POLICY" or "RETURN":
        print("POLICY")