from flask import Flask, request, jsonify
# Assuming trigger_flow is in chat_flow.py and can access the global DataFrames (df, df_customers, etc.)
from chat_flow import trigger_flow 
import pandas as pd
import os

app = Flask(__name__)

# ----------------------------------------------------
# GCS Configuration and Data Loading
# ----------------------------------------------------
# IMPORTANT: This bucket name must match your Cloud Storage bucket
BUCKET_NAME = "group-5-ai-data"
GCS_BASE = f"gs://{BUCKET_NAME}/data"

# Define the full GCS paths for all four files
CLOTH_PARQUET_PATH = f"{GCS_BASE}/cloth_test_set.parquet"
CLOTH_CSV_PATH = f"{GCS_BASE}/cloth_test_set.csv"
CUSTOMERS_CSV_PATH = f"{GCS_BASE}/customers_first_3000.csv"
TRANSACTIONS_CSV_PATH = f"{GCS_BASE}/transactions_first_3000_customers.csv"

# Global DataFrames that will be loaded on startup
# We keep the primary DataFrame named 'df' to minimize changes in 'chat_flow.py'
df = None
df_cloth_csv = None
df_customers = None
df_transactions = None

try:
    print("--- Starting Data Loading from Google Cloud Storage ---")
    
    # 1. Load cloth_test_set.parquet (Original 'df' variable)
    df = pd.read_parquet(CLOTH_PARQUET_PATH)
    print(f"SUCCESS: Loaded cloth_test_set.parquet ({len(df)} rows)")

    # 2. Load customers_first_3000.csv
    df_customers = pd.read_csv(CUSTOMERS_CSV_PATH)
    print(f"SUCCESS: Loaded customers_first_3000.csv ({len(df_customers)} rows)")

    # 3. Load transactions_first_3000_customers.csv
    df_transactions = pd.read_csv(TRANSACTIONS_CSV_PATH)
    print(f"SUCCESS: Loaded transactions_first_3000_customers.csv ({len(df_transactions)} rows)")

    # 4. Load cloth_test_set.csv (if needed by your flow, otherwise this can be removed)
    df_cloth_csv = pd.read_csv(CLOTH_CSV_PATH)
    print(f"SUCCESS: Loaded cloth_test_set.csv ({len(df_cloth_csv)} rows)")

    print("--- All Data Loading Complete ---")

except Exception as e:
    # A critical failure on startup means the app can't function
    print(f"FATAL ERROR: Could not load required data from GCS. Check permissions and paths.")
    print(f"Error details: {e}")
    # You might want to raise the error here to stop the container from starting
    # raise e


# ---------------------------
# API Endpoint
# ---------------------------
@app.route('/query', methods=['POST'])
def handle_query():
    # Check if data was loaded successfully before processing queries
    if df is None:
        return jsonify({"error": "Service not ready. Data failed to load on startup."}), 503
        
    data = request.get_json()

    user_query = data.get('query')
    # Assuming trigger_flow uses the globally loaded DataFrames
    llm_response, images = trigger_flow(user_query)

    return jsonify({
        "response": llm_response,
        "images": images
    }), 200


# ---------------------------
# Run App (Development/Local mode)
# ---------------------------
if __name__ == '__main__':
    # Cloud Run uses the gunicorn/CMD entrypoint, not this block
    # This is only for local testing
    app.run(host="0.0.0.0", port=5000, debug=True)
