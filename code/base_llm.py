import os
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "AIzaSyD06a9SVvel8XDgn02HdKgJGMnSJMS_TrA"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)