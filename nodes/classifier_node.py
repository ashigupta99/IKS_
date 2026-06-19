from google import genai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import os
import json
import numpy as np
from gita_state import GitaState

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# -------------------
# Load data
# -------------------

with open("theme_embeddings.json", "r") as f:
    theme_embeddings = {int(k): v for k, v in json.load(f).items()}

with open("final_grouped_categories.json", "r") as f:
    categories = json.load(f)

# Flatten themes for lookup
themes = []
for broad in categories:
    for sub in broad["subcategories"]:
        themes.append({
            "id": sub["id"],
            "name": sub["name"],
            "description": sub["string"]
        })

# -------------------
# Constants
# -------------------

SIMILARITY_THRESHOLD = 0.4
TOP_N = 5
MIN_THEMES = 2

# -------------------
# Helpers
# -------------------

def get_embedding(text: str) -> list:
    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values


def get_combined_embedding(query: str, rewritten: str) -> np.ndarray:
    # The Gemini SDK call is synchronous and these two embeddings don't
    # depend on each other, so run them concurrently in a thread pool
    # instead of back-to-back. Cuts this node's latency roughly in half.
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_original = executor.submit(get_embedding, query)
        future_rewritten = executor.submit(get_embedding, rewritten)
        emb_original = np.array(future_original.result())
        emb_rewritten = np.array(future_rewritten.result())

    return (emb_original + emb_rewritten) / 2


def classify_themes(query_embedding: np.ndarray) -> list:
    scores = []

    for theme in themes:
        theme_emb = np.array(theme_embeddings[theme["id"]])
        sim = cosine_similarity([query_embedding], [theme_emb])[0][0]
        scores.append((sim, theme["id"], theme["name"]))

    scores.sort(reverse=True)
    top = scores[:TOP_N]

    filtered = [s for s in top if s[0] >= SIMILARITY_THRESHOLD]
    if len(filtered) < MIN_THEMES:
        filtered = top[:MIN_THEMES]

    return [
        {
            "theme_id": tid,
            "theme_name": tname,
            "similarity": round(float(sim), 4)
        }
        for sim, tid, tname in filtered
    ]

# -------------------
# Node
# -------------------

def classifier_node(state: GitaState) -> dict:
    query = state["query"]
    rewritten = state["rewritten_query"]

    combined_emb = get_combined_embedding(query, rewritten)
    matched_themes = classify_themes(combined_emb)
    
    print(f"[classifier_node] matched themes: {[t['theme_name'] for t in matched_themes]}")


    return {
        "combined_embedding": combined_emb.tolist(),
        "matched_themes": matched_themes
    }