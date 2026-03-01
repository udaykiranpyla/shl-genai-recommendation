# generate_submission.py

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ==============================
# LOAD DATA
# ==============================

print("Loading test dataset...")
test_df = pd.read_excel("Gen_AI Dataset.xlsx")  # use your test file if separate

catalog_df = pd.read_csv("shl_catalog_cleaned.csv")
index = faiss.read_index("shl_faiss_index.index")
model = SentenceTransformer("all-MiniLM-L6-v2")

query_col = "Query"

# ==============================
# GENERATE SUBMISSION
# ==============================

submission_rows = []

for _, row in test_df.iterrows():

    query = str(row[query_col])

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding)

    distances, indices = index.search(query_embedding, 10)
    predicted_urls = catalog_df.iloc[indices[0]]["url"].tolist()

    for url in predicted_urls:
        submission_rows.append({
            "Query": query,
            "Assessment_url": url
        })

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv("submission.csv", index=False)

print("✅ submission.csv generated successfully")