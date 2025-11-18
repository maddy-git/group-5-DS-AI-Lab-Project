# Project Overview – AI Stylist

AI Stylist is a Retrieval-Augmented Generation (RAG) based system that improves the performance of Large Language Models (LLMs) using text-based product details extracted from images.  
It embeds product descriptions, stores them using **FAISS**, and retrieves the most relevant data to generate context-aware and accurate recommendations.

---

## Problem Statement
To build a system that recommends products by retrieving relevant text-based information and optimizing responses from large language models.

---

## Usage (from README)
```bash
python3 -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
cd code
python app.py
streamlit run chat_ui.py
```

For API usage (POST `/query`), see `docs/api_doc.md`.

Licensing and citations: see `docs/LICENSES.md`.