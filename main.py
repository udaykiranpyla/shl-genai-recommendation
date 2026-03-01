# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import ast

# ==============================
# CONFIGURATION (SECURE)
# ==============================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-1.5-flash")

# ==============================
# LOAD DATA + MODELS
# ==============================

app = FastAPI()

print("Loading dataset...")
df = pd.read_csv("shl_catalog_cleaned.csv")

print("Loading FAISS index...")
index = faiss.read_index("shl_faiss_index.index")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("System Ready 🚀")

# ==============================
# REQUEST MODEL
# ==============================

class QueryRequest(BaseModel):
    query: str

# ==============================
# HEALTH ENDPOINT
# ==============================

@app.get("/health")
def health_check():
    return {"status": "healthy", "total_assessments": len(df)}

# ==============================
# RECOMMEND ENDPOINT
# ==============================

@app.post("/recommend")
def recommend_assessments(request: QueryRequest):

    query = request.query.strip()

    if not query:
        return {"error": "Query cannot be empty"}

    # STEP 1: Embedding Retrieval
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding)

    distances, indices = index.search(query_embedding, 30)
    candidate_results = df.iloc[indices[0]]

    # STEP 2: LLM Re-ranking
    candidates_text = ""

    for _, row in candidate_results.iterrows():
        candidates_text += f"""
        Name: {row['name']}
        Description: {row['description']}
        URL: {row['url']}
        """

    prompt = f"""
    You are an expert HR assessment recommender.

    Job Description:
    {query}

    Below are candidate SHL assessments.

    {candidates_text}

    Select the 10 MOST relevant assessments.

    Return ONLY a valid Python list of URLs.
    """

    try:
        response = llm.generate_content(prompt)
        selected_urls = ast.literal_eval(response.text.strip())
    except:
        selected_urls = candidate_results["url"].tolist()[:10]

    # STEP 3: Format Response
    final_results = df[df["url"].isin(selected_urls)].head(10)

    recommendations = []

    for _, row in final_results.iterrows():
        recommendations.append({
            "url": row["url"],
            "name": row["name"],
            "adaptive_support": row.get("adaptive_support", "Yes"),
            "description": row.get("description", ""),
            "duration": row.get("duration", ""),
            "remote_support": row.get("remote_support", "Yes"),
            "test_type": ["LLM-ReRanked"]
        })

    return {"recommended_assessments": recommendations}


