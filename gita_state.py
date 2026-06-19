from typing import TypedDict, Optional, List, Any
import numpy as np


class GitaState(TypedDict):
    # Input
    query: str

    # Intent routing
    intent: str                     # counselling | shloka_lookup | shloka_search

    # Counselling pipeline
    rewritten_query: str
    combined_embedding: Optional[List[float]]
    matched_themes: List[dict]
    retrieved_shlokas: List[dict]

    # Output
    response: str
    citations: List[str]
    shlokas_used: List[dict]

    # Critic loop
    critic_passed: bool
    critic_feedback: str
    retry_count: int

    # Memory — list of {query, response} dicts from prior turns
    memory: List[dict]