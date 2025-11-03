# ==========================================================
# 🧩 Feedback-Aware Query Refinement and Re-Retrieval
# Works with existing build_customer_context(), search_products(), recommend_top3_structured()
# ==========================================================

from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import numpy as np
import torch
from sentence_transformers import util, SentenceTransformer

# --- Step 1: Feedback Refinement Prompt ---
feedback_refine_prompt = ChatPromptTemplate.from_template("""
You are a conversational fashion assistant improving recommendations.

You are given:
1. Customer context (demographics, purchase history, etc.)
2. The **previous query** the user asked.
3. The **top 3 products** shown for that query.
4. The **user’s feedback/opinion** about those products.

### TASK
Write a **refined query** for the next product search.
Give importance as follows:
- 60% weight → the user's **current feedback** (what they said they want or dislike)
- 20% weight → the **previous query intent**
- 20% weight → the **customer’s context** and style

### RULES
- Avoid recommending products similar to the rejected ones.
- Focus on what the user now prefers.
- Keep the refined query concise and ready for embedding-based retrieval.
- Return only the refined query, no explanations.

### INPUTS
Customer Context:
age: 50-60
postal_code: Chennai
gender: Female
---------------------------
--- PURCHASE TRANSACTIONS ---
Transaction 935: t_dat: 2018-10-10, article_id: 648414022, sales_channel_id: 2, payment_method: UPI, SKU_No: 648414022.0, prod_return_type: No return, return_Status: Not Returned, delivered: Delivered
---------------------------
--- PRODUCTS PURCHASED ---
Product 2363: SKU_No: 648414022, prod_name: Yate hood, product_type_name: Hoodie, product_group_name: Garment Upper body, graphical_appearance_name: Front print, colour_group_name: Dark Blue, department_name: Jersey Fancy, index_name: Menswear, index_group_name: Menswear, section_name: Contemporary Casual, garment_group_name: Jersey Fancy, detail_desc: Top in sturdy, printed sweatshirt fabric with a lined drawstring hood, kangaroo pocket and ribbing at the cuffs and hem. Soft brushed inside., Eco_Score: 93, Style_Reward_Points: 42, Stock_Status: Low Stock, Price_INR: 3585, Discount_Percentage: 57, Return_Type: No return
---------------------------

Previous Query:
show me something for office wear


Top 3 Products:
```
   SKU_No          prod_name product_type_name product_group_name colour_group_name department_name  Price_INR  Eco_Score  Discount_Percentage    Return_Type  similarity_score
519929006       Topi dress J             Dress  Garment Full body             Black         Dresses       4416         73                   67      No return          0.811020
637268006              April             Skirt Garment Lower body         Dark Blue           Skirt       4093         87                    3      No return          0.812673
686631002  Maggie RW tapered          Trousers Garment Lower body     Greyish Beige         Trouser       3072         68                   17      No return          0.802664

User Feedback:
these are decent but i only want my dress to be suitable for causal waer also if possible

### OUTPUT
Refined Query:
Show me soft, breathable, and comfortable dresses in subtle or dark tones that are suitable for both office and casual wear, ideal for a 50–60-year-old female from Chennai who prefers relaxed fits and smooth fabrics.
""")

feedback_refine_chain = feedback_refine_prompt | llm


# --- Step 2: Function to handle user feedback and get new top 20 ---
def handle_feedback_and_retrieve(
    cust_id: int,
    prev_query: str,
    top3_products_df: pd.DataFrame,
    user_feedback: str,
    top_k: int = 20
):
    """
    1️⃣ Adds previous query, top3 product details, and user feedback into context.
    2️⃣ Uses LLM to create a refined query (weighted 60-20-20).
    3️⃣ Retrieves new top 20 products using existing search_products().
    """

    # --- Build existing context for customer ---
    context = build_customer_context(cust_id)

    # --- Append feedback information ---
    context.append("--- USER FEEDBACK ---")
    context.append(f"Previous Query: {prev_query}")
    context.append("Top 3 Products Displayed:")
    context.append(top3_products_df.to_string(index=False))
    context.append(f"User Feedback: {user_feedback}")
    context.append("---------------------------")

    # --- Generate refined query using feedback prompt ---
    refined = feedback_refine_chain.invoke({
        "context": "\n".join(context),
        "prev_query": prev_query,
        "top3_products": top3_products_df.to_string(index=False),
        "user_feedback": user_feedback
    })
    refined_query = refined.content.strip()
    print(f"\n🧠 Refined Query After Feedback:\n{refined_query}\n")

    # --- Retrieve new top 20 using the refined query ---
    top20_updated = search_products(refined_query, cust_id, top_k=top_k)

    print("\n👕 New Top 20 Recommendations After Feedback:\n")
    print(top20_updated.to_string(index=False))

    return refined_query, top20_updated, top3


# --- Step 3: Example Workflow ---
# Assume you already ran the first 3 code blocks:
# 1️⃣ build_customer_context()
# 2️⃣ search_products()
# 3️⃣ recommend_top3_structured()

cust_id = 5
prev_query = "show me something for office wear"

# Step A — get top 20 & top 3
# --- Step A — get top 20 products ---
# --- Step A — get top 20 products ---
top20 = search_products(prev_query, cust_id, top_k=20)

# --- Step B — call reranker with cust_id (not context) ---
top3_products_text = recommend_top3_structured(prev_query, cust_id, top20)
top3_products_df = top20.head(3)



# Step B — user gives feedback
user_feedback = "These sweaters look too warm and heavy. I want something light and breathable for office wear."

# Step C — refine query & retrieve new top 20
refined_query, new_top20 = handle_feedback_and_retrieve(
    cust_id=cust_id,
    prev_query=prev_query,
    top3_products_df=top3_products_df,
    user_feedback=user_feedback
)
