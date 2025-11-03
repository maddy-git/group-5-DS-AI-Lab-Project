from classify_query import classify_query
from policy_llm import answer_policy_or_return_query
from memory import *
from product_finder_reranker import *

memory = Memory()


def trigger_flow(query):
    if(query.startswith("cust_id:")):
        memory.refresh()
        memory.set_customer(query.split(":")[1])
        return "Logged in successfully"
    elif(query == "logout"):
        memory.refresh()
        return "Logged out successfully"

    llm_response = ""
    classification = classify_query(query, memory.get_history())
    if classification == "POLICY" or classification == "RETURN":
        llm_response = answer_policy_or_return_query(query, memory.get_history())
    elif classification == "PRODUCT_SEARCH":
        top_products = search_products(query, memory.get_customer())
        llm_response = recommend_top3_structured(query, memory.get_customer(), top_products)
    else:
        llm_response = "I am a store assistant which helps you with shopping and store policies.\nPlease ask anything related to these."

    memory.add_chat(query, llm_response)
    return llm_response