# User Documentation (for Non-Technical Users)

## App Overview
The AI Fashion Chatbot acts as your clothing Assistant, designed to handle both customer support inquiries and personalized fashion recommendations.

---
## Key Use Cases
- **Store Policy & Customer Help Desk (Part A)**: Ask questions regarding general store rules, such as return policies, upcoming discounts, or refund timelines.
- **AI Stylist (Part B)**: Get personalized outfit suggestions based on your current request (e.g., "suggest me an outfit for a graduation party"). The agent uses your context (age, gender, purchase history) to refine suggestions. The system is designed to only suggest products available in the company catalogue to prevent irrelevant suggestions.

---
## Input Description
Users interact with the system via a ChatBot UI (implemented using Streamlit).

### Action: Login
Type your customer ID using the format:
```text
cust_id:{customer_id}
```
Valid sample IDs are: `1`, `3`, `5`.

### Action: Logout
Type the following in the chat UI:
```text
logout
```
### Policy Query
Ask a natural language question about store rules (e.g., "What is the refund condition?").

### Product Query
Ask for a recommendation (e.g., "Need a blue jean") or a combination of style and context (e.g., "I need a shirt which goes well with cream pants").
If you are logged out, or if mandatory information (like age and gender) is missing for a product search, the system will pause and ask you a follow-up question before proceeding.

---
## Output Description
The Chatbot's response varies based on the query type:

- **Policy Response**: The system provides a detailed, conversational answer that is grounded directly in the official policy documents.
  - Example: For "what is the return policy," the output includes eligibility, return conditions, non-returnable categories, and the return process.
- **Product Recommendation (AI Stylist)**: The system consistently outputs the top 3 most suitable products in a structured format.
  - Each recommended product will be listed, typically including its: Name, Type, Color, Price (INR), Discount (%), Return Type, and SKU. Product images will be shown along with the text description.

---
## Step-by-Step Instructions
1. **Create a Python virtual environment**
    ```bash
    python3 -m venv .venv
    ```
2. **Activate the virtual environment**
    ```bash
    source ./.venv/bin/activate
    ```
3. **Install requirements**
    ```bash
    pip install -r requirements.txt
    ```
4. **Run Streamlit UI**
    ```bash
    streamlit run code/chat_ui.py
5. **Run Flask backend**
    ```bash
    python code/app.py
    ```

Now the UI will be accessible on http://localhost:8501/ (the port might vary; please refer to the terminal where Streamlit is started).

---

## Screenshots (placeholders)
- Login prompt and successful login confirmation.  
  Placeholder: Add image here (e.g., docs/images/login.png)
- Policy query example and response.  
  Placeholder: Add image here (e.g., docs/images/policy_query.png)
- Product recommendation with images and details.  
  Placeholder: Add image here (e.g., docs/images/product_recs.png)