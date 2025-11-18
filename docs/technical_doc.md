# Technical Documentation – Deployment Details

---
## Platform Used
- **Frontend/UI**: Streamlit
- **Backend/API**: Flask (exposes `/query` endpoint)

---
## Model & Inference
- **Embedders**: Lightweight embedding models are loaded on CPU (consumer-grade capable).
- **LLM**: Google Gemini (Free API tier) is used for generation, retrieval augmentation, and dialogue.

---
## How to Interact
Users primarily interact via the Streamlit ChatBot UI. A backend Flask API is also available.

### Example Commands/Requests
- **Login**: In the Streamlit chat input, type:
  ```text
  cust_id:{customer_id}
  ```
  Example: `cust_id:3`

  Without logging in, the chatbot can still be used, but the LLM may ask for follow-ups like age and gender for product suggestions.
- **Query**: Example prompt:
  ```text
  Need a blue jean
  ```

---
## Local Run Instructions
- Start Streamlit UI:
  ```bash
  streamlit run code/chat_ui.py
  ```
- Start Flask backend:
  ```bash
  python code/app.py
  ```
- Access UI at: `http://localhost:8501/` (port may vary; check the terminal output where Streamlit starts)

---
## Deployment Attempts and Status
We attempted free hosting for end-to-end deployment but encountered memory constraints due to model/data sizes.

### Primary Platform Trial: Render
- **Reason for choice**: Full-stack support, simple workflow, free tier for testing.
- **Frontend (Static Site) – SUCCESS**
  - Live URL: https://group-5-ds-ai-lab-project.onrender.com/
  - Status: Deployment successful, fully functional, responsive, publicly accessible.
- **Backend (API Service) – ATTEMPTED, FAILED ON FREE TIER**
  - Attempt URL: https://group-5-ds-ai-lab-project-1.onrender.com/
  - Status: Service not running
  - Likely cause: High memory requirements from:
    - ML model weights
    - Embedding vectors/indexes
    - Product dataset
    - Additional mapping/preprocessing artifacts
  - Free-tier typical limit: ~512 MB RAM
  - estimated requirement: > 2–4 GB RAM

---
## Recommended Solutions for Backend Deployment
- **Option A: Paid plan on Render**
  - Upgrade to an instance with sufficient RAM (≥4 GB recommended).
- **Option B: Deploy on Google Cloud Platform (GCP)**
  - Viable choices:
    - Compute Engine (VM) – direct control over RAM/CPU, simplest lift-and-shift
    - Cloud Run – containerized deployment; configure memory to meet model + data needs
    - Cloud Functions + Cloud Storage – only if decomposing into microservices
  - By choosing a VM/container with adequate memory, the backend will start reliably.

---
## Conclusion
- **Frontend deployment**: Completed and stable on Render.
- **Backend deployment**: Attempted on Render free tier but failed due to memory limits.
- **Next steps**: Use a paid plan on Render or deploy on GCP with increased RAM to host the backend successfully.