from groq import Groq
from dotenv import load_dotenv
import os
import json
from gita_state import GitaState

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------
# System prompt
# -------------------

CRITIC_PROMPT = """You are a strict quality checker for a Bhagavad Gita counselling assistant.

Evaluate the assistant's response against these criteria:
1. Grounded — does it reference the SPECIFIC TEACHING of each shloka provided, in the response's own words? Paraphrasing the shloka's actual meaning counts as grounded — it does NOT need to closely mirror the shloka's exact wording. It only fails this criterion if the response could apply to ANY situation without these shlokas at all (i.e. generic therapy-speak with no connection to the shloka's specific content).
2. Warm tone — does it sound like a wise friend, not a therapist, not a lecture or a sermon?
3. Actionable — does it end with something useful or a gentle reflective question tied to the shlokas?
4. Not preachy — no moralizing, no "you must", no repeated platitudes.

User's situation:
{query}

Shlokas the response was supposed to be grounded in:
{shloka_context}

Assistant's response:
{response}

Reply with ONLY valid JSON, no markdown fences, in this exact format:
{{"passed": true or false, "feedback": "if failed: name EXACTLY which shloka's teaching was missing or wrong, in one sentence — do not suggest near-verbatim wording, just state what idea was missing. If passed: empty string."}}"""

# -------------------
# Helpers
# -------------------

def build_shloka_context(shlokas: list) -> str:
    lines = []
    for s in shlokas:
        data = s["shloka"]
        sid = data["id"]
        translation = data.get("translation_en", "").strip()
        lines.append(f"[{sid}] {translation}")
    return "\n".join(lines)

# -------------------
# Node
# -------------------

def critic_node(state: GitaState) -> dict:
    query = state["query"]
    response_text = state["response"]
    shlokas_used = state.get("shlokas_used", [])
    retry_count = state.get("retry_count", 0)

    shloka_context = build_shloka_context(shlokas_used)

    prompt = CRITIC_PROMPT.format(
        query=query,
        shloka_context=shloka_context,
        response=response_text
    )

    # 8B model — classification/grading task, doesn't need 70B reasoning,
    # and is faster + cheaper since this node runs on every retry loop.
    result = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150
    )

    raw = result.choices[0].message.content.strip()

    # Strip markdown fences if the model adds them anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        passed = bool(parsed.get("passed", False))
        feedback = parsed.get("feedback", "")
    except Exception:
        # If the critic itself misbehaves, fail open — don't block the user forever
        passed = True
        feedback = ""

    # Lightweight visibility instead of a logging framework — see README
    # note on what proper observability would add here.
    print(f"[critic_node] passed={passed} retry_count={retry_count + 1} feedback={feedback!r}")

    return {
        "critic_passed": passed,
        "critic_feedback": feedback,
        "retry_count": retry_count + 1
    }