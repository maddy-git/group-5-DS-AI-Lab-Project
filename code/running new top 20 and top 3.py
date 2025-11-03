# ==========================================================
# 🎯 Step 5: Full End-to-End Flow — From Feedback to New Top 3
# ==========================================================

# --- Step A: User provides new feedback or query update ---
user_feedback = "These sweaters look too warm and heavy. I want something light and breathable for office wear."
cust_id = 5
prev_query = "show me something for office wear"

# --- Step B: Get initial top 20 + reranked top 3 ---
top20 = search_products(prev_query, cust_id, top_k=20)
top3_products_text = recommend_top3_structured(prev_query, cust_id, top20)
top3_products_df = top20.head(3)

# --- Step C: Handle feedback — refine the query and fetch new top 20 ---
refined_query, new_top20 = handle_feedback_and_retrieve(
    cust_id=cust_id,
    prev_query=prev_query,
    top3_products_df=top3_products_df,
    user_feedback=user_feedback
)


# --- Step D: Re-rank the new top 20 to show top 3 final recommendations ---
new_top3_text = select_top3_after_feedback(
    cust_id=cust_id,
    prev_query=prev_query,
    refined_query=refined_query,
    prev_top3_df=top3_products_df,
    new_top20_df=new_top20
)

# --- Step E: Display the final top 3 results to the user ---
print("\n👗 FINAL TOP 3 PRODUCTS RECOMMENDED AFTER FEEDBACK:\n")
print(new_top3_text)
