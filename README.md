# AI Stylist

*A group project for the DSAI-Lab course under the IIT Madras BS Degree Program.*

---

#DEMO LINK: https://drive.google.com/file/d/1gtPHSeH2PL2NkLydlRJ0gGXaOov4M2Az/view?usp=sharing
---

## Project Title and Overview
AI Stylist is a Retrieval-Augmented Generation (RAG) based system that improves the performance of Large Language Models (LLMs) using text-based product details extracted from images.  
It embeds product descriptions, stores them using **FAISS**, and retrieves the most relevant data to generate context-aware and accurate recommendations.

---

## Problem Statement
To build a system that recommends products by retrieving relevant text-based information and optimizing responses from large language models.

---

## Milestone Documents
- [Milestone 1: Problem Definition & Literature Review]([docs/milestone1_proposal.pdf](https://github.com/maddy-git/group-5-DS-AI-Lab-Project/tree/main/milestones/milestone-1)) - This milestone consists of the problem definition and literature review. The project objective was divided into two parts, A and B. The first one being “Store policy + Customer help desk” and the other part is an “AI Stylist”.
- [Milestone 2: Dataset Preparation]([docs/milestone2_implementation.pdf](https://github.com/maddy-git/group-5-DS-AI-Lab-Project/tree/main/milestones/milestone-2)) – This milestone had details about us choosing a dataset, and performing analysis on it.  
- [Milestone 3: Model Architecture](docs/milestone3_final_report.pdf) – The third milestone had intricate details about the RAG (Retrieval Augmented Generation) method and a handful of embedding models to compare their strengths and weaknesses.
- [Milestone 4: Model Training](https://github.com/maddy-git/group-5-DS-AI-Lab-Project/tree/main/milestones/milestone-4) - This milestone focuses on the implementation of the complete AI Stylist architecture, as outlined in the previously demonstrated flowchart. It builds upon the work from earlier milestones to integrate all components — from data preprocessing to query handling and product recommendation — into a functional end-to-end system.

---

## Repository Structure
```
AI-Stylist/
├── code
│ ├── adding_context.py
│ ├── app.py
│ ├── base_llm.py
│ ├── chat_flow.py
│ ├── chat_ui.py
│ ├── classify_query.py
│ ├── constants.py
│ ├── content
│ │ └── faq_and_policy.pdf
│ ├── embed_policy.py
│ ├── gorq_query.py
│ ├── memory.py
│ ├── policy_llm.py
│ ├── product_finder_reranker.py
│ ├── product_search.py
│ ├── pycache
│ │ ├── app.cpython-310.pyc
│ │ ├── chat_flow.cpython-310.pyc
│ │ ├── classify_query.cpython-310.pyc
│ │ └── memory_init.cpython-310.pyc
│ └── user_context_checker.py
├── milestones
│ ├── milestone-1
│ │ └── Milestone 1.pdf
│ ├── milestone-2
│ │ ├── articles.csv
│ │ ├── customers.csv
│ │ ├── EDA_on_catalog.ipynb
│ │ ├── faq_and_policy.pdf
│ │ ├── Milestone-2.pdf
│ │ ├── purchase_history.xlsx
│ │ └── sample_images
│ │ ├── 0108775015.jpg
│ │ ├── 0108775044.jpg
│ │ ├── 0108775051.jpg
│ │ ├── 0282832015.jpg
│ │ ├── 0282832017.jpg
│ │ └── 0282832022.jpg
│ └── milestone-3
│ ├── EmbeddingTest.ipynb
│ └── Milestone-3.pdf
├── README.md
└── requirements.txt
```

---

## Usage Instructions

```bash
python3 -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
cd code
python app.py
streamlit run chat_ui.py
```

---

### To Login, type this in the chat input:
```
cust_id:{customer_id}
```

---
### To Logout, type this in the chat input:
```
logout
```

---

## Reproducibility Setup

Dataset: Text-based data for catalog, customers, transactions, and image data, both from Kaggle. 
Link: (https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/)

Environment: Python 3.9+, dependencies in requirements.txt.

Reproducibility: Use the same embeddings and FAISS index.

Notebooks: Notebooks are not needed for running it.

---

## Team Members and Roles
Image data management - Neeraj and Harsha

Policy RAG and streamlit UI - Madhav and Manish

Customer context building - Harsha and Manish

Product RAG - Harsha and Manish

Product Embedding - Neeraj and Adarsh

Memory and Customer session Management - Manish

Preparing the submission document - Adarsh, Manish and Harsha

---

## License
a) Project License: Educational use under IIT Madras BS Program.
b) Dataset Credits and License: Product data belongs to respective sources; used for academic purposes only.

---

## Acknowledgements
Developed as part of the Data Science and Artificial Intelligence degree-level course at IIT Madras.
