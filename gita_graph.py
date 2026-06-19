"""
gita_graph.py  —  LangGraph pipeline for the Bhagavad Gita counselling assistant.

Graph structure:
  START
    └── intent_node
          ├── counselling → rewrite_node → classifier_node → retriever_node
          │                                                 → generator_node → critic_node
          │                                                                        ├── pass  → END
          │                                                                        └── fail  → generator_node (max 2 retries)
          ├── shloka_lookup → lookup_node → END
          └── shloka_search → search_node → END
"""

from langgraph.graph import StateGraph, END
from gita_state import GitaState

from nodes.intent_node     import intent_node
from nodes.rewrite_node    import rewrite_node
from nodes.classifier_node import classifier_node
from nodes.retriever_node  import retriever_node
from nodes.generator_node  import generator_node
from nodes.critic_node     import critic_node
from nodes.lookup_node     import lookup_node
from nodes.search_node     import search_node


CRISIS_PHRASES = [
    "end my life", "ending my life", "kill myself", "don't want to live",
    "want to die", "suicide", "nothing feels worth it", "no point in living"
]

def is_crisis(query: str) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in CRISIS_PHRASES)


def route_intent(state: GitaState) -> str:
    return state["intent"]  # "counselling" | "shloka_lookup" | "shloka_search"


def route_critic(state: GitaState) -> str:
    if state["critic_passed"] or state["retry_count"] >= 2:
        return "end"
    return "retry"


def build_graph():
    g = StateGraph(GitaState)

    g.add_node("intent_node",     intent_node)
    g.add_node("rewrite_node",    rewrite_node)
    g.add_node("classifier_node", classifier_node)
    g.add_node("retriever_node",  retriever_node)
    g.add_node("generator_node",  generator_node)
    g.add_node("critic_node",     critic_node)
    g.add_node("lookup_node",     lookup_node)
    g.add_node("search_node",     search_node)

    g.set_entry_point("intent_node")

    g.add_conditional_edges(
        "intent_node",
        route_intent,
        {
            "counselling":   "rewrite_node",
            "shloka_lookup": "lookup_node",
            "shloka_search": "search_node",
        }
    )

    g.add_edge("rewrite_node",    "classifier_node")
    g.add_edge("classifier_node", "retriever_node")
    g.add_edge("retriever_node",  "generator_node")
    g.add_edge("generator_node",  "critic_node")

    g.add_conditional_edges(
        "critic_node",
        route_critic,
        {
            "end":   END,
            "retry": "generator_node",
        }
    )

    g.add_edge("lookup_node", END)
    g.add_edge("search_node", END)

    return g.compile()


gita_graph = build_graph()


def run(query: str, memory: list = None) -> GitaState:
    if is_crisis(query):
        return {
            "response": "What you're carrying sounds incredibly heavy, and I'm glad you said it out loud.\n\nPlease reach out to iCall right now — they're free, confidential, and speak Hindi and English:\n📞 9152987821\n\nThe Gita's wisdom is here for you, but this moment calls for a real human voice first.",
            "intent": "crisis",
            "shlokas_used": [],
            "citations": [],
            "retry_count": 0,
            "critic_passed": True,
            "critic_feedback": ""
        }
    initial_state: GitaState = {
        "query":              query,
        "intent":             "",
        "rewritten_query":    "",
        "combined_embedding": None,
        "matched_themes":     [],
        "retrieved_shlokas":  [],
        "response":           "",
        "citations":          [],
        "shlokas_used":       [],
        "critic_passed":      False,
        "critic_feedback":    "",
        "retry_count":        0,
        "memory":             memory or [],
    }
    return gita_graph.invoke(initial_state)