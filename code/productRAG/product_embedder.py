from sentence_transformers import SentenceTransformer
from . import constants
import pandas as pd
import numpy as np

cloth_data = pd.read_csv(constants.product_dir)

model = SentenceTransformer("BAAI/bge-small-en")

text_columns = [
    "prod_name", "product_type_name", "product_group_name",
    "graphical_appearance_name", "colour_group_name", "department_name",
    "index_name", "index_group_name", "section_name", "garment_group_name",
    "detail_desc"
]

cloth_data["embed_text"] = cloth_data[text_columns].astype(str).agg("|".join, axis=1)
embeddings = model.encode(cloth_data["embed_text"].tolist(), show_progress_bar=True)


cloth_data["embeddings"] = embeddings.tolist()
cloth_data.to_parquet(constants.product_embedding_dir, index=False)

print("✅ Embedding completed!")
print("Shape of embeddings:", np.array(embeddings).shape)