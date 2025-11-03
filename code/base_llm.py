import os
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "AIzaSyBfCV-TobP0xCdhrLvcZwbjSrIWjo03CX8"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)