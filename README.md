# AI Stylist
------------------------------------------------------------------
## A group project on Data Science and Artificial Intelligence for the IIT Madras BS Degree program's degree-level course with the same name.
------------------------------------------------------------------
### This repo consists of a RAG (Retrieval Augmented Generation) on an image dataset to optimize the output of existing LLMs (large Language Models).
### It does this by first embedding the textual data of the images using an embedding model, and feeding it to the FAISS (Facebook AI Similarity Search).
------------------------------------------------------------------
### Steps to run:-
`command 1`


pip install -r requirements.txt
streamlit run chat_ui.py
python app.py

To login just type cust_id:{customer_id}
To logout just type logout