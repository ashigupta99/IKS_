import re
from groq import Groq
from dotenv import load_dotenv
import os
import json
from gita_state import GitaState

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------
# Load corpus
# -------------------

with open("corpus_raw.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

corpus_map = {s["id"]: s for s in corpus}

# -------------------
# Prompt
# -------------------

EXPLAIN_PROMPT = """Explain this Bhagavad Gita shloka clearly and briefly (100-150 words).
Cover: what it means, who says it, and why it matters.

Shloka {sid}: {translation}
Commentary: {commentary}"""

# -------------------
# Helpers
# -------------------

def extract_shloka_id(query: str) -> str | None:
    # Primary: direct dotted form, e.g. "2.47"
    match = re.search(r'\b(\d+)\.(\d+)\b', query)
    if match:
        return f"{match.group(1)}.{match.group(2)}"

    # Fallback: "chapter 2 ... shlok/shloka/verse 47" (any order, words in between)
    match = re.search(
        r'chap(?:ter)?\.?\s*(\d+).{0,30}?(?:shlok(?:a)?|verse)\s*(?:no\.?|number)?\s*(\d+)',
        query,
        re.IGNORECASE
    )
    if match:
        return f"{match.group(1)}.{match.group(2)}"

    # Fallback: reversed order — "shlok/verse 47 ... chapter 2"
    match = re.search(
        r'(?:shlok(?:a)?|verse)\s*(?:no\.?|number)?\s*(\d+).{0,30}?chap(?:ter)?\.?\s*(\d+)',
        query,
        re.IGNORECASE
    )
    if match:
        return f"{match.group(2)}.{match.group(1)}"

    return None

# -------------------
# Node
# -------------------

def lookup_node(state: GitaState) -> dict:
    query = state["query"]

    sid = extract_shloka_id(query)
    if not sid:
        return {
            "response": "I couldn't find a shloka number in your question. You can ask like '2.47' or 'chapter 2 shlok 47'.",
            "citations": [],
            "shlokas_used": []
        }

    shloka = corpus_map.get(sid)

    if not shloka:
        return {
            "response": f"Shloka {sid} isn't in the corpus — double-check the chapter and verse number.",
            "citations": [],
            "shlokas_used": []
        }

    sanskrit = shloka.get("shloka_devanagari", "N/A")
    translation = shloka.get("translation_en", "N/A")
    bhashya = shloka.get("bhashya_hindi", "").strip()

    prompt = EXPLAIN_PROMPT.format(
        sid=sid,
        translation=translation,
        commentary=bhashya[:300] if bhashya else "None"
    )

    result = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300
    )
    explanation = result.choices[0].message.content.strip()

    # Build the full response text — Sanskrit + translation + commentary (if any) + explanation
    parts = [
        f"**Shloka {sid}**",
        f"Sanskrit: {sanskrit}",
        f"Translation: {translation}",
    ]
    if bhashya:
        parts.append(f"Shankara Bhashya: {bhashya}")
    else:
        parts.append("(No Shankara commentary for this shloka)")
    parts.append(f"\n{explanation}")

    response_text = "\n\n".join(parts)

    return {
        "response": response_text,
        "citations": [sid],
        "shlokas_used": [{"shloka": shloka}]
    }