# --- Step 4️⃣: Auto-Select Top 3 From Updated Top-20 After Feedback ---
def select_top3_after_feedback(
    cust_id: int,
    prev_query: str,
    refined_query: str,
    prev_top3_df: pd.DataFrame,
    new_top20_df: pd.DataFrame
):
    """
    Uses the LLM reranker to select top 3 products from the new top 20 results
    after user feedback refinement.

    Inputs:
      - cust_id: current customer ID
      - prev_query: previous user query before feedback
      - refined_query: query refined based on feedback
      - prev_top3_df: previous top 3 product DataFrame (for reference)
      - new_top20_df: new 20-product DataFrame retrieved via embeddings

    Output:
      - String (LLM structured table with top 3)
    """

    # ✅ Build the updated context for this customer
    context = build_customer_context(cust_id)
    context.append("--- FEEDBACK CONTEXT ---")
    context.append(f"Previous Query: {prev_query}")
    context.append("Previous Top 3 Products:")
    context.append(prev_top3_df.to_string(index=False))
    context.append(f"Refined Query: {refined_query}")
    context.append("---------------------------")

    # ✅ Convert top 20 dataframe to formatted string for reranker
    top_20_str = new_top20_df.to_string(index=False)

    # ✅ Define rerank prompt (same priority rules)
    feedback_rerank_prompt = ChatPromptTemplate.from_template("""
You are an expert fashion recommendation AI.
Given the customer’s context, the previous and refined queries, and the new top 20 retrieved products,
select the **top 3** that best align with the refined query (user’s latest feedback and intent).

### PRIORITIES:
- 60% → Current refined query intent (based on feedback)
- 20% → Previous query meaning
- 10% → Customer context
- 10% → Previous top 3 products (style continuity only)

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


Previous Top 3 Products
```
   SKU_No          prod_name product_type_name product_group_name colour_group_name department_name  Price_INR  Eco_Score  Discount_Percentage    Return_Type  similarity_score
519929006       Topi dress J             Dress  Garment Full body             Black         Dresses       4416         73                   67      No return          0.811020
637268006              April             Skirt Garment Lower body         Dark Blue           Skirt       4093         87                    3      No return          0.812673
686631002  Maggie RW tapered          Trousers Garment Lower body     Greyish Beige         Trouser       3072         68                   17      No return          0.802664

User Feedback:
these are decent but i only want my dress to be suitable for causal waer also if possible

Refined Query:
Show me soft, breathable, and comfortable dresses in subtle or dark tones, especially dark blue, suitable for both office and casual wear. These should be ideal for a 50-60-year-old female from Chennai who prefers relaxed fits and smooth fabrics, similar to the comfort of her previously purchased 'Yate hood' hoodie, and ideally with a high Eco-Score



New Top 20 Products:
👕 New Top 20 Recommendations After Feedback:

   SKU_No               prod_name product_type_name product_group_name colour_group_name  department_name  Price_INR  Eco_Score  Discount_Percentage    Return_Type  similarity_score
648414022               Yate hood            Hoodie Garment Upper body         Dark Blue     Jersey Fancy       3585         93                   57      No return          0.862019
648414004               Yate hood            Hoodie Garment Upper body             White     Jersey Fancy       3982         79                   59 15 days return          0.855183
649223002     Nice throw on dress             Dress  Garment Full body         Dark Blue            Dress       1039         87                   63      No return          0.852276
642938001     RELAXED SKINNY HANA          Trousers Garment Lower body              Blue Young Girl Denim       2159         89                   28      No return          0.851110
610747002                   Smile           Sweater Garment Upper body             Black   Knitwear Basic       1799         90                   26  7 days return          0.850816
610543011           SPEED Paprika           Sweater Garment Upper body              Grey           Jersey       2533         63                   53 15 days return          0.849504
642960002             Polly Dress             Dress  Garment Full body        Light Blue  Jersey Occasion        546         86                   41 15 days return          0.848591
663944004                 Serrano             Dress  Garment Full body         Dark Blue            Dress       1125         54                   22 15 days return          0.848369
590007003              Hijack Top               Top Garment Upper body         Dark Blue   Woven Occasion       2675         91                    5 15 days return          0.848186
519929006            Topi dress J             Dress  Garment Full body             Black          Dresses       4416         73                   67      No return          0.848008
610543009           SPEED Paprika           Sweater Garment Upper body             Black           Jersey        665         60                   34 15 days return          0.847731
510398013               Yate hood            Hoodie Garment Upper body        Dark Beige     Jersey Fancy       4010         98                   53 15 days return          0.847164
399061008             Jacket Slim            Jacket Garment Upper body        Light Blue          Outwear       1298         50                   60  7 days return          0.846519
669216002                   Polly            Hoodie Garment Upper body    Greenish Khaki     Jersey Basic       1155         68                   69      No return          0.846519
601468007             Bossy dress             Dress  Garment Full body       Dark Orange   Woven Occasion       4840         82                   56 15 days return          0.846371
617919021                  Melody             Dress  Garment Full body         Dark Blue            Dress       4088         71                   56 15 days return          0.846202
588771001       Lilac smock dress             Dress  Garment Full body             Black            Dress       1793         90                    7      No return          0.846181
663944001                 Serrano             Dress  Garment Full body        Light Grey            Dress        566         69                   65  7 days return          0.845529
629824001 POLLY daycare dress TVP             Dress  Garment Full body              Blue  Baby Girl Woven       3410         97                   52 15 days return          0.845378
686631003       Maggie RW tapered          Trousers Garment Lower body   Yellowish Brown          Trouser       4220         80                   60 15 days return          0.844926

### OUTPUT
Return only the top 3 product rows as a formatted table
(including all columns, aligned spacing — no explanation text).
""")

    feedback_rerank_chain = feedback_rerank_prompt | llm

    # ✅ Invoke reranker LLM
    response = feedback_rerank_chain.invoke({
        "context": "\n".join(context),
        "prev_query": prev_query,
        "refined_query": refined_query,
        "prev_top3": prev_top3_df.to_string(index=False),
        "new_top20": top_20_str
    })

    print("\n🎯 Final Updated Top 3 After Feedback:\n")
    print(response.content)
    return response.content
