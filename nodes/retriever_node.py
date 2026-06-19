import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gita_state import GitaState

# -------------------
# Load data
# -------------------

with open("theme_to_shlokas_reranked.json", "r", encoding="utf-8") as f:
    theme_to_shlokas = {item["theme_id"]: item["shlokas"] for item in json.load(f)}

with open("updated_corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

corpus_map = {s["id"]: s for s in corpus}

with open("shloka_embeddings.json", "r", encoding="utf-8") as f:
    shloka_embeddings = json.load(f)

# -------------------
# Constants
# -------------------

FINAL_TOP_N = 6
NON_TEACHING_SHLOKAS = {
    "2.8", "1.47", "2.1", "2.2",
    "2.11", "2.26", "2.27", "2.28",
    "18.60", "16.11", "16.12", "16.18",
    "9.12", "2.7" 
}

# -------------------
# Helpers
# -------------------

def collect_shlokas(matched_themes: list) -> dict:
    """
    For each matched theme, fetch its top 5 shlokas from reranked JSON.
    Deduplicate — same shloka can appear in multiple themes.
    Track which themes each shloka came from.
    """
    shloka_pool = {}

    for theme in matched_themes:
        tid = theme["theme_id"]
        theme_sim = theme["similarity"]

        if tid not in theme_to_shlokas:
            continue

        for entry in theme_to_shlokas[tid]:
            sid = entry["id"]
            rank = entry["rank"]

            if sid not in corpus_map:
                continue

            # Skip shlokas without commentary (Chapter 1 — narrative, no teaching value)
            bhashya = corpus_map[sid].get("bhashya_hindi", "").strip()
            if not bhashya or "did not comment" in bhashya.lower():
                continue

            # Skip lament-only shlokas — emotionally resonant but no actual teaching
            if sid in NON_TEACHING_SHLOKAS:
                continue

            if sid not in shloka_pool:
                shloka_pool[sid] = {
                    "shloka": corpus_map[sid],
                    "themes_matched": [],
                    "best_rank": rank
                }

            shloka_pool[sid]["themes_matched"].append({
                "theme_id": tid,
                "theme_name": theme["theme_name"],
                "theme_similarity": round(theme_sim, 4),
                "rank_in_theme": rank
            })

            if rank < shloka_pool[sid]["best_rank"]:
                shloka_pool[sid]["best_rank"] = rank

    return shloka_pool


def rerank_shlokas(shloka_pool: dict, query_embedding: np.ndarray) -> list:
    """
    Rerank all collected shlokas against the combined query embedding.
    """
    scored = []

    for sid, data in shloka_pool.items():
        if sid not in shloka_embeddings:
            continue

        shloka_emb = np.array(shloka_embeddings[sid])
        sim = cosine_similarity([query_embedding], [shloka_emb])[0][0]

        scored.append({
            "id": sid,
            "query_similarity": round(float(sim), 4),
            "themes_matched": data["themes_matched"],
            "shloka": data["shloka"]
        })

    scored.sort(key=lambda x: x["query_similarity"], reverse=True)
    return scored[:FINAL_TOP_N]

# -------------------
# Node
# -------------------

def retriever_node(state: GitaState) -> dict:
    matched_themes = state["matched_themes"]
    query_embedding = np.array(state["combined_embedding"])

    shloka_pool = collect_shlokas(matched_themes)
    final_shlokas = rerank_shlokas(shloka_pool, query_embedding)
    print(f"[retriever_node] top shlokas: {[s['id'] for s in final_shlokas]}")

    return {"retrieved_shlokas": final_shlokas}