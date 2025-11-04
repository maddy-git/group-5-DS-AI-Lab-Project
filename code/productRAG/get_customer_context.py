import pandas as pd
from . import constants

def build_customer_context(cust_id: int):
    # --- Load required datasets ---
    customers = pd.read_csv(constants.customers_dir)
    transactions = pd.read_csv(constants.transactions_dir)
    cloth_data = pd.read_csv(constants.product_dir)

    # --- Ensure data consistency ---
    customers["cust_id"] = pd.to_numeric(customers["cust_id"], errors="coerce").astype("Int64")
    transactions["cust_id"] = pd.to_numeric(transactions["cust_id"], errors="coerce").astype("Int64")

    # --- Initialize context ---
    context = []

    # --- CUSTOMER INFO ---
    # ✅ Ensure cust_id type matches dataset column
    cust_id = int(cust_id)
    customer_ids = customers["cust_id"].dropna().astype(int).values

    if cust_id not in customer_ids:
        return f"❌ Customer ID {cust_id} not found in dataset."

    customer_info = customers[customers["cust_id"] == cust_id].iloc[0]

    # Exclude unnecessary columns
    excluded_cols = ["size", "cust_id", "embed_text"]

    context.append("--- CUSTOMER CONTEXT ---")
    for col, val in customer_info.items():
        if col not in excluded_cols:
            context.append(f"{col}: {val}")
    context.append("---------------------------")

    # --- TRANSACTIONS ---
    customer_transactions = transactions[transactions["cust_id"] == cust_id]
    context.append("--- PURCHASE TRANSACTIONS ---")

    if customer_transactions.empty:
        context.append("No transactions found for this customer.")
    else:
        for i, row in customer_transactions.iterrows():
            tx_details = ", ".join([
                f"{col}: {row[col]}" for col in customer_transactions.columns if col != "cust_id"
            ])
            context.append(f"Transaction {i}: {tx_details}")

    context.append("---------------------------")

    # --- PRODUCTS PURCHASED ---
    purchased_skus = customer_transactions["SKU_No"].unique()
    purchased_products = cloth_data[cloth_data["SKU_No"].isin(purchased_skus)]

    context.append("--- PRODUCTS PURCHASED ---")

    if purchased_products.empty:
        context.append("No product details found for this customer's purchases.")
    else:
        for i, row in purchased_products.iterrows():
            prod_details = ", ".join([f"{col}: {row[col]}" for col in purchased_products.columns])
            context.append(f"Product {i}: {prod_details}")

    context.append("---------------------------")

    print(f"✅ Context successfully created for customer {cust_id}")

    formatted_context = ""
    for cont in context:
        formatted_context = formatted_context + cont + "\n"

    print(formatted_context)
    return formatted_context