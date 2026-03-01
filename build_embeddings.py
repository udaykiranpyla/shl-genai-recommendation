# build_embeddings.py

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("shl_catalog_full.csv")

print("Total records:", len(df))

# Combine text fields for better embedding
df["combined_text"] = (
    df["name"].fillna("") + " " +
    df["description"].fillna("")
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

embeddings = model.encode(
    df["combined_text"].tolist(),
    show_progress_bar=True
)

# Convert to numpy
embeddings = np.array(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, "shl_faiss_index.index")

# Save dataframe (cleaned)
df.to_csv("shl_catalog_cleaned.csv", index=False)

print("FAISS index saved as shl_faiss_index.index")
print("Clean dataset saved as shl_catalog_cleaned.csv")