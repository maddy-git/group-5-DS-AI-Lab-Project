import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_core.messages import HumanMessage, AIMessage
from productRAG import findKBest, reranker, constants

customer_context1 = '''
--- CUSTOMER CONTEXT ---
age: 25-30
postal_code: Mumbai
gender: Male
---------------------------
--- PURCHASE TRANSACTIONS ---
Transaction 735: t_dat: 2018-09-29, SKU_No: 666448006.0, payment_method: UPI, return_Status: Not Returned, delivered: Delivered
Transaction 1561: t_dat: 2018-09-27, SKU_No: 583558001.0, payment_method: Card, return_Status: Returned, delivered: Delivered
Transaction 3221: t_dat: 2018-09-21, SKU_No: 640244003.0, payment_method: Card, return_Status: Not Returned, delivered: Delivered
---------------------------
--- PRODUCTS PURCHASED ---
Product 1272: prod_name: Noel denim dress, product_type_name: Dress, product_group_name: Garment Full body, graphical_appearance_name: Denim, colour_group_name: Blue, department_name: Dresses, index_name: Divided, index_group_name: Divided, section_name: Divided Collection, garment_group_name: Dresses Ladies, detail_desc: Short, fitted dress in stretch denim with a collar, zip down the front, chest pockets and long sleeves with press-studs at the cuffs. Unlined., Eco_Score: 61, Style_Reward_Points: 62, Stock_Status: Out of Stock, Price_INR: 2954, Discount_Percentage: 61, Return_Type: 15 days return
Product 2237: prod_name: CORY CORD SKIRT, product_type_name: Skirt, product_group_name: Garment Lower body, graphical_appearance_name: Solid, colour_group_name: Black, department_name: Skirts, index_name: Divided, index_group_name: Divided, section_name: Divided Collection, garment_group_name: Skirts, detail_desc: Short 5-pocket skirt in cotton corduroy with a zip fly and button., Eco_Score: 86, Style_Reward_Points: 98, Stock_Status: Low Stock, Price_INR: 3917, Discount_Percentage: 10, Return_Type: No return
Product 2867: prod_name: Janet sweater, product_type_name: Sweater, product_group_name: Garment Upper body, graphical_appearance_name: Solid, colour_group_name: Blue, department_name: Knitwear, index_name: Ladieswear, index_group_name: Ladieswear, section_name: Womens Everyday Collection, garment_group_name: Knitwear, detail_desc: Long-sleeved jumper in a soft, fine knit with ribbing around the neckline, cuffs and hem., Eco_Score: 94, Style_Reward_Points: 16, Stock_Status: Low Stock, Price_INR: 3120, Discount_Percentage: 10, Return_Type: 7 days return
---------------------------
'''

customer_context2 = '''
--- CUSTOMER CONTEXT ---
age: 30-35
postal_code: Bengaluru
gender: Male
---------------------------
'''

customer_context3 = '''
--- CUSTOMER CONTEXT ---
age: 50-60
postal_code: Chennai
gender: Female
---------------------------
--- PURCHASE TRANSACTIONS ---
Transaction 935: t_dat: 2018-10-10, SKU_No: 648414022.0, payment_method: UPI, return_Status: Not Returned, delivered: Delivered
---------------------------
--- PRODUCTS PURCHASED ---
Product 2363: prod_name: Yate hood, product_type_name: Hoodie, product_group_name: Garment Upper body, graphical_appearance_name: Front print, colour_group_name: Dark Blue, department_name: Jersey Fancy, index_name: Menswear, index_group_name: Menswear, section_name: Contemporary Casual, garment_group_name: Jersey Fancy, detail_desc: Top in sturdy, printed sweatshirt fabric with a lined drawstring hood, kangaroo pocket and ribbing at the cuffs and hem. Soft brushed inside., Eco_Score: 93, Style_Reward_Points: 42, Stock_Status: Low Stock, Price_INR: 3585, Discount_Percentage: 57, Return_Type: No return
---------------------------
'''

customer_context4 = '''
--- CUSTOMER CONTEXT ---
age: 60-70
postal_code: Jaipur
gender: Female
---------------------------
--- PURCHASE TRANSACTIONS ---
Transaction 3443: t_dat: 2018-10-28, SKU_No: 626813002.0, payment_method: UPI, return_Status: Not Returned, delivered: Delivered
Transaction 4391: t_dat: 2018-10-28, SKU_No: 626813004.0, payment_method: UPI, return_Status: Not Returned, delivered: Delivered
---------------------------
--- PRODUCTS PURCHASED ---
Product 1866: prod_name: Syrup ISW 19, product_type_name: Blouse, product_group_name: Garment Upper body, graphical_appearance_name: All over pattern, colour_group_name: White, department_name: Blouse, index_name: Ladieswear, index_group_name: Ladieswear, section_name: Womens Everyday Collection, garment_group_name: Blouses, detail_desc: Blouse in a viscose crêpe weave with a V-neck, short sleeves and rounded hem. Slightly longer at the back. Unlined., Eco_Score: 97, Style_Reward_Points: 41, Stock_Status: In Stock, Price_INR: 822, Discount_Percentage: 11, Return_Type: No return
Product 1867: prod_name: Syrup ISW 19, product_type_name: Blouse, product_group_name: Garment Upper body, graphical_appearance_name: All over pattern, colour_group_name: White, department_name: Blouse, index_name: Ladieswear, index_group_name: Ladieswear, section_name: Womens Everyday Collection, garment_group_name: Blouses, detail_desc: Blouse in a viscose crêpe weave with a V-neck, short sleeves and rounded hem. Slightly longer at the back. Unlined., Eco_Score: 63, Style_Reward_Points: 57, Stock_Status: In Stock, Price_INR: 1387, Discount_Percentage: 6, Return_Type: No return
---------------------------
'''

customer_context5 = '''
--- CUSTOMER CONTEXT ---
age: 40-50
postal_code: Bengaluru
gender: Female
---------------------------
--- PURCHASE TRANSACTIONS ---
Transaction 187: t_dat: 2018-10-11, SKU_No: 668537001.0, payment_method: UPI, return_Status: Not Returned, delivered: Delivered
Transaction 1218: t_dat: 2018-10-27, SKU_No: 633226004.0, payment_method: UPI, return_Status: Not Returned, delivered: Delivered
Transaction 1248: t_dat: 2018-10-27, SKU_No: 612509004.0, payment_method: UPI, return_Status: Not Returned, delivered: Delivered
Transaction 2267: t_dat: 2018-10-01, SKU_No: 536933002.0, payment_method: UPI, return_Status: Not Returned, delivered: Delivered
Transaction 2503: t_dat: 2018-10-11, SKU_No: 667378001.0, payment_method: Cash, return_Status: Returned, delivered: Delivered
Transaction 2552: t_dat: 2018-10-27, SKU_No: 727263002.0, payment_method: UPI, return_Status: nan, delivered: Not Delivered (Wrong Item)
Transaction 3468: t_dat: 2018-10-01, SKU_No: 563519001.0, payment_method: UPI, return_Status: Not Returned, delivered: Delivered
Transaction 3535: t_dat: 2018-10-02, SKU_No: 688873008.0, payment_method: Cash, return_Status: Not Returned, delivered: Delivered
Transaction 4538: t_dat: 2018-10-27, SKU_No: 643361001.0, payment_method: UPI, return_Status: Not Returned, delivered: Delivered
Transaction 5545: t_dat: 2018-10-01, SKU_No: 663016003.0, payment_method: Card, return_Status: Not Returned, delivered: Delivered
---------------------------
--- PRODUCTS PURCHASED ---
Product 626: prod_name: Reggie, product_type_name: Top, product_group_name: Garment Upper body, graphical_appearance_name: Solid, colour_group_name: Light Beige, department_name: Jersey fancy, index_name: Ladieswear, index_group_name: Ladieswear, section_name: Womens Everyday Collection, garment_group_name: Jersey Fancy, detail_desc: Long-sleeved top in jersey made from Tencel™️ lyocell with a small stand-up collar., Eco_Score: 77, Style_Reward_Points: 31, Stock_Status: In Stock, Price_INR: 1413, Discount_Percentage: 15, Return_Type: No return
Product 957: prod_name: Simba, product_type_name: Sweater, product_group_name: Garment Upper body, graphical_appearance_name: Solid, colour_group_name: Black, department_name: Jersey Basic, index_name: Ladieswear, index_group_name: Ladieswear, section_name: Womens Everyday Basics, garment_group_name: Jersey Basic, detail_desc: Fitted, polo-neck top in soft viscose jersey., Eco_Score: 52, Style_Reward_Points: 97, Stock_Status: In Stock, Price_INR: 4821, Discount_Percentage: 0, Return_Type: No return
Product 1553: prod_name: Burt Skinny Trs, product_type_name: Trousers, product_group_name: Garment Lower body, graphical_appearance_name: Solid, colour_group_name: Dark Orange, department_name: Trouser, index_name: Menswear, index_group_name: Menswear, section_name: Men Suits & Tailoring, garment_group_name: Trousers, detail_desc: Suit trousers in a stretch weave made from a cotton blend with a concealed hook-and-eye fastener, zip fly, side pockets, jetted back pockets and legs with creases. Skinny Fit – a fit with slightly shorter legs that is close-fitting at the thighs, knees and ankles to create a completely fitted silhouette., Eco_Score: 73, Style_Reward_Points: 20, Stock_Status: Low Stock, Price_INR: 2008, Discount_Percentage: 40, Return_Type: 15 days return
Product 2010: prod_name: Zion tartan slim blz, product_type_name: Blazer, product_group_name: Garment Upper body, graphical_appearance_name: Check, colour_group_name: Black, department_name: Blazer, index_name: Menswear, index_group_name: Menswear, section_name: Men Suits & Tailoring, garment_group_name: Dressed, detail_desc: Single-breasted jacket in woven fabric containing some wool with narrow notch lapels, a decorative buttonhole and two buttons at the front. Chest pocket, flap front pockets and two inner pockets, one with a button. Decorative buttons at the cuffs and a single back vent. Lined. Slim fit that tapers at the chest and waist, which combined with slightly narrower sleeves creates a fitted silhouette., Eco_Score: 97, Style_Reward_Points: 40, Stock_Status: In Stock, Price_INR: 1414, Discount_Percentage: 54, Return_Type: 15 days return
Product 2297: prod_name: Andy check skinny blz, product_type_name: Blazer, product_group_name: Garment Upper body, graphical_appearance_name: Check, colour_group_name: Black, department_name: Blazer, index_name: Menswear, index_group_name: Menswear, section_name: Men Suits & Tailoring, garment_group_name: Dressed, detail_desc: Single-breasted jacket in a stretch weave with narrow notch lapels and a decorative buttonhole. Open chest pocket, flap front pockets and three inner pockets, one with a button. One button at the front, decorative buttons at the cuffs and a single back vent. Lined. Skinny fit – a slightly shorter style that shapes in at the chest and tapers sharply at the waist. This combined with narrow shoulders and sleeves creates a completely fitted silhouette., Eco_Score: 99, Style_Reward_Points: 65, Stock_Status: In Stock, Price_INR: 3318, Discount_Percentage: 52, Return_Type: 7 days return
Product 2758: prod_name: Whisper, product_type_name: Sweater, product_group_name: Garment Upper body, graphical_appearance_name: Melange, colour_group_name: Light Pink, department_name: Knitwear, index_name: Ladieswear, index_group_name: Ladieswear, section_name: Womens Everyday Collection, garment_group_name: Knitwear, detail_desc: Wide polo-neck jumper in a soft rib knit containing some wool with low dropped shoulders and long, wide sleeves with ribbed cuffs. Slightly longer at the back., Eco_Score: 50, Style_Reward_Points: 63, Stock_Status: Out of Stock, Price_INR: 2385, Discount_Percentage: 29, Return_Type: No return
Product 2873: prod_name: Tulip, product_type_name: Trousers, product_group_name: Garment Lower body, graphical_appearance_name: Solid, colour_group_name: Black, department_name: Knitwear Basic, index_name: Ladieswear, index_group_name: Ladieswear, section_name: Womens Everyday Basics, garment_group_name: Knitwear, detail_desc: Fitted calf-length skirt in a rib knit with an elasticated waist., Eco_Score: 68, Style_Reward_Points: 89, Stock_Status: In Stock, Price_INR: 3115, Discount_Percentage: 19, Return_Type: 15 days return
Product 2903: prod_name: Doutzen, product_type_name: Jacket, product_group_name: Garment Upper body, graphical_appearance_name: Solid, colour_group_name: Dark Beige, department_name: Outwear, index_name: Ladieswear, index_group_name: Ladieswear, section_name: Womens Jackets, garment_group_name: Outdoor, detail_desc: Long coat in pile with narrow notch lapels, a concealed press-stud fastening at the front and pockets in the side seams. Lined., Eco_Score: 61, Style_Reward_Points: 28, Stock_Status: Low Stock, Price_INR: 2427, Discount_Percentage: 49, Return_Type: No return
Product 3353: prod_name: Gyda!, product_type_name: Blouse, product_group_name: Garment Upper body, graphical_appearance_name: All over pattern, colour_group_name: Black, department_name: Blouse, index_name: Ladieswear, index_group_name: Ladieswear, section_name: Womens Tailoring, garment_group_name: Blouses, detail_desc: Blouse in a soft weave with a narrow collar, concealed buttons down the front, long sleeves with buttoned cuffs and a rounded hem., Eco_Score: 74, Style_Reward_Points: 77, Stock_Status: Low Stock, Price_INR: 4434, Discount_Percentage: 31, Return_Type: 15 days return
Product 3587: prod_name: Pila Blouse, product_type_name: Top, product_group_name: Garment Upper body, graphical_appearance_name: All over pattern, colour_group_name: Dark Orange, department_name: Woven Occasion, index_name: Ladieswear, index_group_name: Ladieswear, section_name: Womens Premium, garment_group_name: Dresses Ladies, detail_desc: V-neck blouse in airy, patterned chiffon with a pleated collar and gathers at the shoulders. Decorative covered buttons down the front, a seam at the waist with elastication at the back, and a frill trim at the hem. Long puff sleeves with covered buttons at the cuffs., Eco_Score: 54, Style_Reward_Points: 89, Stock_Status: Out of Stock, Price_INR: 3256, Discount_Percentage: 42, Return_Type: 7 days return
---------------------------
'''

history1 = [
    HumanMessage(content="I need shirt"),
    AIMessage(content="\nProduct 1:\n• Name: 3 PK V-NECK SS BASIC\n• Type: T-shirt\n• Color: Dark Blue\n• Price: 1300 INR\n• Discount: 64%\n• Return Type: 7 days return\n• SKU: N/A\n\nProduct 2:\n• Name: Ronny R-Neck\n• Type: T-shirt\n• Color: Dark Blue\n• Price: 4032 INR\n• Discount: 24%\n• Return Type: No return\n• SKU: N/A\n\nProduct 3:\n• Name: Gary Tee\n• Type: T-shirt\n• Color: White\n• Price: 2398 INR\n• Discount: 0%\n• Return Type: 15 days return\n• SKU: N/A")
]

history2 = [
    HumanMessage(content="Hi, can i get a pant"),
    AIMessage(content="Please help me with your age and gender to assist you better"),
    HumanMessage(content="My age is 24 and my gender is Male")
]

history3 = [
    HumanMessage(content="Can you recommend a laptop?"),
    AIMessage(content="I am a store assistant which helps you with shopping and store policies.\nPlease ask anything related to these."),
    HumanMessage(content="I need a pant"),
    AIMessage(content="\nProduct 1:\n• Name: Montana pants\n• Type: Trousers\n• Color: Dark Blue\n• Price: 3920 INR\n• Discount: 13%\n• Return Type: No return\n• SKU: N/A\n\nProduct 2:\n• Name: Slim Straight 5pkt Midway 1\n• Type: Trousers\n• Color: Black\n• Price: 2323 INR\n• Discount: 67%\n• Return Type: 7 days return\n• SKU: N/A\n\nProduct 3:\n• Name: Jerry jogger bottoms\n• Type: Trousers\n• Color: Dark Blue\n• Price: 2847 INR\n• Discount: 35%\n• Return Type: 7 days return\n• SKU: N/A")
]

history4 = [
    HumanMessage(content="Hi, can i get a top"),
    AIMessage(content="Please help me with your age and gender to assist you better")
]


#Query 1
query1 = "What about a red one?"
query2 = "help me find a blue one"
query3 = "I need blue jean at low price around 2000 INR"
query4 = "My age is 21 and my gender is Female. Help me with a red top"
query5 = "I need a shirt which goes well with cream pant, while creating refined query explicitly mention the short which will look good"
query6 = "show me formal shirts that would go well for office meetings"
query7 = "suggest some casual wear options like polos or chinos for everyday use"
query8 = "show me something stylish yet comfortable for casual outings"
query9 = "I’m looking for a light summer dress suitable for both office and evening wear"
query10 = "recommend me some trendy tops in pastel colors for weekend brunches"



file_name = "test/results.txt"
# Open a file and write text
with open(file_name, "a") as file:
    file.write("\nTemp: " + str(constants.temp) + "\n")
    file.write("Top-K: " + str(constants.top_k) + "\n")
    file.write("Refine Prompt: " + str(constants.refine_prompt_name) + "\n")
    file.write("Rerank Prompt: " + str(constants.reranker_prompt_name) + "\n")

    file.write("Query1: \n")
    top_products = findKBest.search_products(query1, customer_context1, history1, constants.top_k)
    llm_response = reranker.recommend_top3_structured(query1, customer_context1, history1, top_products)
    file.write(llm_response)
    file.write("\n_______________________\n")

    file.write("Query2: \n")
    top_products = findKBest.search_products(query2, customer_context2, history2, constants.top_k)
    llm_response = reranker.recommend_top3_structured(query2, customer_context2, history2, top_products)
    file.write(llm_response)
    file.write("\n_______________________\n")

    file.write("Query3: \n")
    top_products = findKBest.search_products(query3, customer_context1, history1, constants.top_k)
    llm_response = reranker.recommend_top3_structured(query3, customer_context1, history1, top_products)
    file.write(llm_response)
    file.write("\n_______________________\n")

    file.write("Query4: \n")
    top_products = findKBest.search_products(query4, customer_context3, history3, constants.top_k)
    llm_response = reranker.recommend_top3_structured(query4, customer_context3, history3, top_products)
    file.write(llm_response)
    file.write("\n_______________________\n")

    file.write("Query5: \n")
    top_products = findKBest.search_products(query5, customer_context2, history2, constants.top_k)
    llm_response = reranker.recommend_top3_structured(query5, customer_context2, history2, top_products)
    file.write(llm_response)
    file.write("\n_______________________\n")

    file.write("Query6: \n")
    top_products = findKBest.search_products(query6, customer_context2, history2, constants.top_k)
    llm_response = reranker.recommend_top3_structured(query6, customer_context2, history2, top_products)
    file.write(llm_response)
    file.write("\n_______________________\n")

    file.write("Query7: \n")
    top_products = findKBest.search_products(query7, customer_context2, history1, constants.top_k)
    llm_response = reranker.recommend_top3_structured(query7, customer_context2, history1, top_products)
    file.write(llm_response)
    file.write("\n_______________________\n")

    file.write("Query8: \n")
    top_products = findKBest.search_products(query8, customer_context5, history4, constants.top_k)
    llm_response = reranker.recommend_top3_structured(query8, customer_context5, history4, top_products)
    file.write(llm_response)
    file.write("\n_______________________\n")

    file.write("Query9: \n")
    top_products = findKBest.search_products(query9, customer_context3, history4, constants.top_k)
    llm_response = reranker.recommend_top3_structured(query9, customer_context3, history4, top_products)
    file.write(llm_response)
    file.write("\n_______________________\n")

    file.write("Query10: \n")
    top_products = findKBest.search_products(query10, customer_context3, history4, constants.top_k)
    llm_response = reranker.recommend_top3_structured(query10, customer_context3, history4, top_products)
    file.write(llm_response)
    file.write("\n_______________________\n")

    file.write("========================================================================")
    file.write("========================================================================")