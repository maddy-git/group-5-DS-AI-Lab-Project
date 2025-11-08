from productRAG import classify_query, get_customer_context, findKBest, reranker, image_fetcher
from policy_llm import answer_policy_or_return_query
from memory import *
from user_context_checker import handle_user_query
import re

memory = Memory()


def trigger_flow(query):
    print("Memory:")
    print(memory.get_history())

    #LOGIN LOGOUT
    if(query.startswith("cust_id:")):
        memory.refresh()
        memory.set_customer(query.split(":")[1])
        return "Logged in successfully", None
    elif(query == "logout"):
        memory.refresh()
        return "Logged out successfully", None

    #CHECKING AGE AND GENDER INFO
    if(memory.get_customer() == -1):
        response= handle_user_query(query, [], memory)
        if("age" in response.lower() or "gender" in response.lower()):
            memory.add_chat(query, response)
            return response, None

    #RAG
    llm_response = ""
    images = []
    classification = classify_query.classify_query(query, memory.get_history())
    if classification == "POLICY" or classification == "RETURN":
        llm_response = answer_policy_or_return_query(query, memory.get_history())
    elif classification == "PRODUCT_SEARCH":
        if(memory.get_customer_context() == ""):
            customer_context = get_customer_context.build_customer_context(memory.get_customer())
            memory.set_customer_context(customer_context)
        else:
            customer_context = memory.get_customer_context()
        top_products = findKBest.search_products(query, customer_context, memory.get_history())
        llm_response = reranker.recommend_top3_structured(query, customer_context, memory.get_history(), top_products)
        sku_list = re.findall(r'SKU:\s*(\d+)', llm_response)
        images = [image_fetcher.get_image_path(sku) for sku in sku_list]

    else:
        llm_response = "I am a store assistant which helps you with shopping and store policies.\nPlease ask anything related to these."

    memory.add_chat(query, llm_response)
    return llm_response, images