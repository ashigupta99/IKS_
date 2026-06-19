from groq import Groq
from dotenv import load_dotenv
import os
from gita_state import GitaState

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

REWRITE_PROMPT = """You are helping a counselling system grounded in the Bhagavad Gita.

A user has shared their problem. Rewrite it in philosophical and emotional language that captures:
- The core emotional state (anxiety, grief, anger, confusion, attachment, etc.)
- The underlying life situation (duty, relationships, loss, identity, purpose, etc.)
- Any spiritual or existential dimension if present

Keep it concise (2-3 sentences). Do not add Gita references. Just rewrite the human problem in deeper language.

User query: {query}

Rewritten query:"""


def rewrite_node(state: GitaState) -> dict:
    query = state["query"]

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}]
    )

    rewritten = response.choices[0].message.content.strip()
    return {"rewritten_query": rewritten}