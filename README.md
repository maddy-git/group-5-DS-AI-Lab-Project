# AI Stylist

*A group project for the DSAI-Lab course project under IIT Madras BS Degree Program.*  

---

## Overview
AI Stylist uses **Retrieval-Augmented Generation (RAG)** to enhance LLM outputs by retrieving text-based product details derived from product images.  
It embeds product descriptions, stores them in **FAISS**, and retrieves the most relevant ones for better, context-aware responses.

**Output:** Personalized clothing suggestions or product matches based on user queries.

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run chat_ui.py
python app.py
```

### To Login
```
cust_id:{customer_id}
```

### To Logout
```
logout
```

### Tech Stack

Python

Streamlit

Flask

FAISS

Embedding Model (e.g., Sentence Transformers)

LLM (e.g., GPT / Gemini)


## Note

This project is part of the IIT Madras BS in Data Science and Applications course.
