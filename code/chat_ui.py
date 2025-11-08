import streamlit as st
import requests

st.title("👗 AI Fashion Chatbot")

user_input = st.text_input("Ask your question:", placeholder="e.g., What goes well with a red shirt for summer?")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        res = requests.post("http://127.0.0.1:5000/query", json={"query": user_input})
        if res.status_code == 200:
            answer = res.json().get("response", "")
            print(res.json())
            st.text(f"{answer}")

            images = res.json().get("images", "")
            if images:
                for image in images:
                    st.image(image)

        else:
            st.error("Something went wrong. Check backend logs.")
