from groq import Groq
from dotenv import load_dotenv
import os
import json
from gita_state import GitaState

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------
# Load QA lookup
# -------------------

with open("shloka_qa.json", "r", encoding="utf-8") as f:
    shloka_to_qa = json.load(f)

# -------------------
# System prompt
# -------------------

SYSTEM_PROMPT = """You are a compassionate counsellor grounded in the wisdom of the Bhagavad Gita. You speak with warmth, clarity, and depth — never preachy, never distant.

STRICT RULES:
- You MUST reference the specific teaching of each primary shloka provided. Name what the shloka teaches (not the verse number — the actual meaning in plain English).
- Do NOT use generic therapeutic language like "I can sense your fear", "it's as if you're standing before...", "your heart is overwhelmed", or any metaphor you invented. These are not from the Gita.
- Every paragraph must trace back to a specific shloka's teaching. If you cannot connect a statement to a shloka provided, do not say it.
- If the provided shlokas do not fully address the person's situation, acknowledge that honestly rather than inventing guidance.
- Do not quote Sanskrit. Express the shloka's meaning in plain English.
- 18.66 means surrender of ego and attachment to outcomes — never paraphrase it as 'give up' or 'abandon duties'.

Your style:
- Briefly acknowledge the person's situation (1 sentence only — plain, not poetic)
- Then immediately bring in what the Gita teaches through the provided shlokas
- Speak as a wise friend who has read the Gita, not a therapist or a scholar
- Keep responses focused: 150-200 words
- Do not moralize or lecture
- End with one gentle, specific question that invites reflection — tied to what the shlokas reveal

Here are examples of the tone and style to follow:
{few_shot}
"""

RETRY_SUFFIX = """

IMPORTANT — Previous attempt was rejected by the quality checker for this reason:
{critic_feedback}

Fix the above issue in this attempt. Do not repeat the same mistake.
Be more specific — name the actual teaching of each shloka directly in your response."""

# -------------------
# Helpers
# -------------------

FEW_SHOT_PER_SHLOKA = 1
MAX_FEW_SHOT = 4

def build_few_shot(shlokas: list) -> str:
    examples = []
    seen_questions = set()

    for s in shlokas:
        sid = s["shloka"]["id"]
        pairs = shloka_to_qa.get(sid, [])

        for pair in pairs[:FEW_SHOT_PER_SHLOKA]:
            q = pair["question"]
            if q in seen_questions:
                continue
            seen_questions.add(q)
            examples.append(f"User: {q}\nAssistant: {pair['answer']}")

        if len(examples) >= MAX_FEW_SHOT:
            break

    if not examples:
        return "No examples available."

    return "\n\n".join(f"EXAMPLE {i+1}\n{ex}" for i, ex in enumerate(examples))


def build_shloka_context(shlokas: list) -> str:
    lines = []
    for s in shlokas:
        data = s["shloka"]
        sid = data["id"]
        translation = data.get("translation_en", "").strip()
        bhashya = data.get("bhashya_hindi", "").strip()

        block = f"[{sid}] {translation}"
        if bhashya:
            block += f"\nCommentary: {bhashya[:300]}"
        lines.append(block)

    return "\n\n".join(lines)


def build_memory_context(memory: list) -> str:
    if not memory:
        return ""
    lines = ["Previous conversation turns:"]
    for turn in memory[-3:]:  # last 3 turns max
        lines.append(f"User: {turn['query']}")
        lines.append(f"Assistant: {turn['response']}\n")
    return "\n".join(lines)

# -------------------
# Node
# -------------------

def generator_node(state: GitaState) -> dict:
    query = state["query"]
    shlokas = state["retrieved_shlokas"]
    memory = state.get("memory", [])
    critic_feedback = state.get("critic_feedback", "")
    retry_count = state.get("retry_count", 0)

    primary = shlokas[:3]
    supporting = shlokas[3:]

    few_shot = build_few_shot(shlokas)
    system_prompt = SYSTEM_PROMPT.format(few_shot=few_shot)

    # On retry, append critic feedback to system prompt
    if retry_count > 0 and critic_feedback:
        system_prompt += RETRY_SUFFIX.format(critic_feedback=critic_feedback)

    shloka_context = build_shloka_context(primary)
    supporting_context = build_shloka_context(supporting) if supporting else ""
    memory_context = build_memory_context(memory)

    user_message = ""

    if memory_context:
        user_message += f"{memory_context}\n\n"

    user_message += f"""The person has shared:
"{query}"

Primary shlokas to draw from (you MUST address each one's specific teaching in your response):
{shloka_context}"""

    if supporting_context:
        user_message += f"""

Additional shlokas (use only if directly relevant):
{supporting_context}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=500
    )

    answer = response.choices[0].message.content.strip()
    citations = [s["shloka"]["id"] for s in primary]

    return {
        "response": answer,
        "citations": citations,
        "shlokas_used": primary
    }