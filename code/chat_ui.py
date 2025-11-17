import streamlit as st
import requests

st.title("👗 AI Fashion Chatbot (RAG + Groq)")

user_input = st.text_input("Ask your question:", placeholder="e.g., What goes well with a red shirt for summer?")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        res = requests.post(" https://myfashion-481842609060.us-central1.run.app", json={"query": user_input})
        if res.status_code == 200:
            answer = res.json().get("response", "")
            print(res.json())
            st.markdown(f"**AI Stylist:** {answer}")
        else:
            st.error("Something went wrong. Check backend logs.")
