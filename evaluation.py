# evaluation.py

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import ast

# ==============================
# CONFIGURE GEMINI
# ==============================

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-1.5-flash")

# ==============================
# LOAD DATA
# ==============================

print("Loading training dataset...")
train_df = pd.read_excel("Gen_AI Dataset.xlsx")

print("Columns:", train_df.columns)

catalog_df = pd.read_csv("shl_catalog_cleaned.csv")
index = faiss.read_index("shl_faiss_index.index")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# HELPER FUNCTIONS
# ==============================

def extract_slug(url):
    url = str(url).strip().lower().rstrip("/")
    return url.split("/")[-1]

def recall_at_k(predicted, actual, k=10):
    predicted_top_k = predicted[:k]
    hits = len(set(predicted_top_k) & set(actual))
    if len(actual) == 0:
        return 0
    return hits / len(actual)

# ==============================
# COLUMN DETECTION
# ==============================

query_col = "Query"
url_col = "Assessment_url"

# ==============================
# CALCULATE RECALL WITH LLM
# ==============================

recall_scores = []

for _, row in train_df.iterrows():

    query = str(row[query_col])
    actual_urls = str(row[url_col]).split(",")
    actual_slugs = [extract_slug(u) for u in actual_urls]

    # Step 1: Retrieve top 30 using embeddings
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding)

    distances, indices = index.search(query_embedding, 30)
    candidate_results = catalog_df.iloc[indices[0]]

    # Step 2: Prepare candidates for LLM
    candidates_text = ""
    for _, r in candidate_results.iterrows():
        candidates_text += f"""
        Name: {r['name']}
        Description: {r['description']}
        URL: {r['url']}
        """

    prompt = f"""
    You are an expert HR assessment recommender.

    Job Description:
    {query}

    From the following assessments, select the 10 MOST relevant ones.

    Return ONLY a Python list of URLs.
    """

    try:
        response = llm.generate_content(prompt + candidates_text)
        selected_urls = ast.literal_eval(response.text.strip())
    except:
        selected_urls = candidate_results["url"].tolist()[:10]

    predicted_slugs = [extract_slug(u) for u in selected_urls]

    recall = recall_at_k(predicted_slugs, actual_slugs, 10)
    recall_scores.append(recall)

mean_recall = sum(recall_scores) / len(recall_scores)

print("\n🔥 Improved Mean Recall@10:", round(mean_recall, 4))