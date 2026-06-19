import re
from groq import Groq
from dotenv import load_dotenv
import os
from gita_state import GitaState

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

INTENT_PROMPT = """You are an intent classifier for a Bhagavad Gita assistant.

Classify the user query into exactly one of these intents:
- counselling      : user is sharing a personal problem, emotion, or life situation and needs guidance
- shloka_lookup    : user is asking about a specific shloka by chapter/verse number (e.g. "explain 2.47", "what is shloka 1.1")
- shloka_search    : user is asking which shloka says something, or searching for a verse by content/topic (e.g. "which shloka talks about duty", "where does Krishna say about anger")

Reply with ONLY one word: counselling, shloka_lookup, or shloka_search.

Query: {query}
Intent:"""


def intent_node(state: GitaState) -> dict:
    query = state["query"]

    # Fast regex check for explicit shloka ID like 2.47
    if re.search(r'\b\d+\.\d+\b', query):
        return {"intent": "shloka_lookup"}

    # 8B model — single-word classification, doesn't need 70B reasoning,
    # and keeps this node fast since it runs on every single query.
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": INTENT_PROMPT.format(query=query)}],
        max_tokens=10,
        temperature=0
    )
    intent = response.choices[0].message.content.strip().lower()

    if intent not in ("counselling", "shloka_lookup", "shloka_search"):
        intent = "counselling"

    return {"intent": intent}