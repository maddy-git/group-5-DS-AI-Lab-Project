import streamlit as st
import requests
import json
import re

def parse_products(text: str):
    products_list = []
    blocks = re.split(r"Product \d+:", text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        product_dict = {}
        pattern = r"•\s*(.+?):\s*(.+)"

        for key, value in re.findall(pattern, block):
            # Remove trailing commas from values like "SKU: 558961001,"
            product_dict[key.strip()] = value.strip().rstrip(",")

        products_list.append(product_dict)

    return products_list

# 🎨 Streamlit Configuration
st.set_page_config(layout="wide")

st.title("👗 AI Fashion Chatbot")



# --- Chat Interface ---
user_input = st.text_input("Ask your question:", placeholder="e.g., What goes well with a red shirt for summer?")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Ask"):
    with st.spinner("Thinking..."):
        res = requests.post("http://127.0.0.1:5000/query", json={"query": user_input})
        if res.status_code == 200:
            answer = res.json().get("response", "")
            print(res.json().get("response", ""))
            if answer:
                st.write(f"AI Stylist:{answer}")

                        st.markdown(f"**{product.get('Name', 'N/A')}**")
                        st.markdown(f"Type: **{product.get('Type', 'N/A')}**")
                        st.markdown(f"Color: **{product.get('Color', 'N/A')}**")
                        st.markdown(f"Price: **{product.get('Price', 'N/A')}**")
                        st.markdown(f"Discount: **{product.get('Discount', 'N/A')}**")
                        st.markdown(f"Return Type: **{product.get('Return Type', 'N/A')}**")
                        st.markdown(f"SKU No: **{product.get('SKU', 'N/A')}**")

                        st.markdown('<button type="button">🛒 Add to Cart</button>', unsafe_allow_html=True)
        else:
            st.markdown("---")
            st.info(f"**AI Response:** {answer}")
