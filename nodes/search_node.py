from google import genai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import numpy as np
from gita_state import GitaState

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

with open("corpus_raw.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

corpus_map = {s["id"]: s for s in corpus}

with open("shloka_embeddings.json", "r", encoding="utf-8") as f:
    shloka_embeddings = json.load(f)

TOP_N = 8  # fetch more to account for filtered-out results

NON_TEACHING_SHLOKAS = {
    "2.8", "1.47", "2.1", "2.2", "2.11", "2.26",
    "2.27", "2.28", "18.60", "16.11", "16.12",
    "16.18", "9.12", "2.7"
}

def search_node(state: GitaState) -> dict:
    query = state["query"]

    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
    query_emb = np.array(response.embeddings[0].values)

    scored = []
    for sid, emb in shloka_embeddings.items():
        sim = cosine_similarity([query_emb], [np.array(emb)])[0][0]
        scored.append((float(sim), sid))

    scored.sort(reverse=True)

    shlokas_used = []
    result_lines = ["Here are the most relevant shlokas I found:\n"]

    for sim, sid in scored:
        if len(shlokas_used) >= 5:
            break

        if sid in NON_TEACHING_SHLOKAS:
            continue

        shloka = corpus_map.get(sid, {})
        translation = shloka.get("translation_en", "").strip()
        bhashya = shloka.get("bhashya_hindi", "").strip()

        # skip shlokas with no real content in either field
        if "did not comment" in translation.lower():
            continue
        if not bhashya or "did not comment" in bhashya.lower():
            continue

        shlokas_used.append({"shloka": shloka})
        result_lines.append(
            f"**[{sid}]** (relevance: {round(sim, 4)})\n{translation[:250]}"
        )

    response_text = "\n\n".join(result_lines)

    return {
        "response": response_text,
        "citations": [item["shloka"]["id"] for item in shlokas_used],
        "shlokas_used": shlokas_used
    }