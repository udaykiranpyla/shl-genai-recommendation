from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
import os
import ast
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

app = FastAPI()

# ==============================
# Secure API Key
# ==============================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-1.5-flash")

# ==============================
# Load Only Lightweight Things
# ==============================

df = pd.read_csv("shl_catalog_cleaned.csv")
index = faiss.read_index("shl_faiss_index.index")

model = None  # Lazy loading

# ==============================
# Request Model
# ==============================

class QueryRequest(BaseModel):
    query: str

# ==============================
# Health Endpoint
# ==============================

@app.get("/health")
def health():
    return {"status": "healthy", "total_assessments": len(df)}

# ==============================
# Recommendation Endpoint
# ==============================

@app.post("/recommend")
def recommend(request: QueryRequest):

    global model

    # Load model only when first request comes
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    query = request.query.strip()

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding)

    distances, indices = index.search(query_embedding, 20)
    candidate_results = df.iloc[indices[0]]

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

    Select the 10 MOST relevant assessments.

    Return ONLY a Python list of URLs.
    """

    try:
        response = llm.generate_content(prompt + candidates_text)
        selected_urls = ast.literal_eval(response.text.strip())
    except:
        selected_urls = candidate_results["url"].tolist()[:10]

    final_results = df[df["url"].isin(selected_urls)].head(10)

    return {
        "recommended_assessments": final_results.to_dict(orient="records")
    }