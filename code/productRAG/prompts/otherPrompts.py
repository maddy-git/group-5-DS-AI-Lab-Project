from langchain_core.prompts import ChatPromptTemplate

structured_response_prompt = ChatPromptTemplate.from_template("""
You are a helpful fashion assistant that structures product data neatly for the customer.

You are given a raw text table of the **Top 3 Recommended Products**.
Extract and display the key product details in a clear, readable format.

### Extract and show:
- Product Name
- Product Type
- Color
- Price (in INR)
- Discount (%)
- Return Type
- SKU Number

### Format Example:
Product 1:
• Name: ...
• Type: ...
• Color: ...
• Price: ...
• Discount: ...
• Return Type: ...
• SKU: ...

### RAW INPUT
{raw_top3}

### OUTPUT
Return only the structured formatted product list (no explanations, no extra text).
""")