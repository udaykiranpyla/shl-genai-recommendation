# search_engine.py

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load data
df = pd.read_csv("shl_catalog_cleaned.csv")

# Load FAISS index
index = faiss.read_index("shl_faiss_index.index")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

def search_assessments(query, top_k=10):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = df.iloc[indices[0]]

    return results[["name", "url"]]

if __name__ == "__main__":
    query = input("Enter job description or query: ")
    results = search_assessments(query)

    print("\nTop Recommendations:\n")
    print(results)