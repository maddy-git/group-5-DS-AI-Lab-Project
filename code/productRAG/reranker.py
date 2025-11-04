from . import constants

structured_response_chain = constants.structured_response_prompt | constants.llm


rerank_chain = constants.reranker_prompt | constants.llm


def recommend_top3_structured(customer_query, customer_context, chat_history, top_products):
    top_20_str = top_products.to_string(index=False)

    response = rerank_chain.invoke({
        "customer_context": customer_context,
        "chat_history": chat_history,
        "query": customer_query,
        "top_20": top_20_str
    })

    raw_top3_text = response.content.strip()
    structured_output = structured_response_chain.invoke({
        "raw_top3": raw_top3_text
    })

    # ✅ Step 5: Clean escaped newline characters and extra spaces
    formatted_output = structured_output.content.replace("\\n", "\n").replace("\\t", "\t").strip()
    print(formatted_output)
    # ✅ Return only formatted structured summary
    return f"\n{formatted_output}"