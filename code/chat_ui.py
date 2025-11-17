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

if st.button("Ask", type="primary"):
    if not user_input:
        st.warning("Please enter a question first.")
        # Stop processing if no input
        st.stop()

    answer = ""
    products = []

    with st.spinner("Talking to AI..."):
        try:
            # 🎯 Connect to the Flask backend on localhost
            res = requests.post(" https://myfashion-481842609060.us-central1.run.app", json={"query": user_input})

            if res.status_code == 200:
                data = res.json()

                # 1. Extract the plain text response from the 'response' key
                answer = data.get("response")
                if ("Product 1" in answer):
                    products = parse_products(answer)
                    images = data.get("images", [])
                    for i in range(len(images)):
                        products[i]['url'] = images[i]

            else:
                st.error(f"Backend call failed. Status code: {res.status_code}. Response: {res.text}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend server. Please ensure your Flask app is running")
        except json.JSONDecodeError:
            st.error("Failed to parse JSON response from backend. Check the server's output.")

        if products:
            st.subheader("🛍️ Recommended Styles")

            # Use 3 columns for the product grid
            num_cols = 3
            cols = st.columns(num_cols)

            for i, product in enumerate(products):
                # Cycle through the columns for a grid layout
                with cols[i % num_cols]:
                    with st.container(border=True):

                        try:
                            st.image(
                                product.get("url", ""),
                                caption=product.get("name", "Product"),
                                use_container_width=True
                            )
                        except Exception as e:
                            print("Unable to fetch image")

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
