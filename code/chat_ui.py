import streamlit as st
import requests

st.title("👗 AI Fashion Chatbot")

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

            images = res.json().get("images", "")
            if images:
                for image in images:
                    st.image(image)

        else:
            st.error("Something went wrong. Check backend logs.")
