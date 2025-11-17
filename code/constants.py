# ----------------------------------------------------
# GCS Configuration
# ----------------------------------------------------
# IMPORTANT: This bucket name must match your Cloud Storage bucket
BUCKET_NAME = "group-5-ai-data"
GCS_BASE = f"gs://{BUCKET_NAME}/data"

# Define the full GCS paths for all data files
CLOTH_PARQUET_PATH = f"{GCS_BASE}/cloth_test_set.parquet"
CLOTH_CSV_PATH = f"{GCS_BASE}/cloth_test_set.csv"
CUSTOMERS_CSV_PATH = f"{GCS_BASE}/customers_first_3000.csv"
TRANSACTIONS_CSV_PATH = f"{GCS_BASE}/transactions_first_3000_customers.csv"

# Path for the product embeddings file
PRODUCT_EMBEDDING_PATH = f"{GCS_BASE}/cloth_products_bge_embedded.parquet"


# ----------------------------------------------------
# DataFrame Dictionary Keys for Consistent Access
# ----------------------------------------------------
# These keys should be used to access dataframes in the dictionary passed to trigger_flow
DF_CLOTH_PARQUET_KEY = 'df_cloth_parquet' # The original 'df'
DF_CLOTH_CSV_KEY = 'df_cloth_csv'
DF_CUSTOMERS_KEY = 'df_customers'
DF_TRANSACTIONS_KEY = 'df_transactions
