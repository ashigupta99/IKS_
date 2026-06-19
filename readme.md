# 🪷 Bhagavad Gita Counselling Assistant

A conversational AI that offers guidance grounded in the Bhagavad Gita — built on a multi-node LangGraph pipeline with semantic theme classification, reranked shloka retrieval, and a critic-in-the-loop quality check.

---

## What it does

- **Counselling** — user shares a personal problem; the system finds the most relevant Gita teachings and responds with warmth and specificity
- **Shloka lookup** — user asks about a specific verse (`2.47`, `chapter 6 shlok 35`); returns Sanskrit, translation, Shankara Bhashya, and an explanation
- **Shloka search** — user asks which verse talks about a topic; returns top semantic matches from the corpus
- **Crisis intercept** — detects suicidal ideation before the pipeline runs and returns a human crisis resource (iCall India) instead of philosophical guidance

---

## Architecture

```
START
└── intent_node
      ├── counselling → rewrite_node → classifier_node → retriever_node
      │                                                 → generator_node → critic_node
      │                                                                        ├── pass → END
      │                                                                        └── fail → generator_node (max 1 retry)
      ├── shloka_lookup → lookup_node → END
      └── shloka_search → search_node → END
```

### Nodes

| Node | Model | Purpose |
|---|---|---|
| `intent_node` | Llama 3.1 8B (Groq) | Classifies query as counselling / lookup / search |
| `rewrite_node` | Llama 3.3 70B (Groq) | Rewrites query in philosophical/emotional language for better embedding match |
| `classifier_node` | Gemini embedding-001 | Embeds original + rewritten query, finds top-5 matching themes from 75 subcategories |
| `retriever_node` | — | Pools shlokas from matched themes, filters non-teaching verses, reranks by query similarity |
| `generator_node` | Llama 3.3 70B (Groq) | Generates counselling response grounded in retrieved shlokas, with few-shot QA examples |
| `critic_node` | Llama 3.3 70B (Groq) | Checks grounding, tone, actionability, and non-preachiness; triggers retry if failed |
| `lookup_node` | Llama 3.3 70B (Groq) | Explains a specific shloka by ID |
| `search_node` | Gemini embedding-001 | Semantic search over full corpus by topic |

---

## Data

| File | Description |
|---|---|
| `corpus_raw.json` | All 700+ shlokas with Sanskrit, English translation, Shankara Bhashya |
| `updated_corpus.json` | Corpus with corrected translations (e.g. 18.45 fix) |
| `final_grouped_categories.json` | 75 subcategory themes across 10 broad categories |
| `theme_to_shlokas_reranked.json` | Top-8 shlokas per theme, manually audited and corrected |
| `shloka_embeddings.json` | Gemini embeddings for every shloka (pre-computed) |
| `theme_embeddings.json` | Gemini embeddings for every theme description (pre-computed) |
| `shloka_qa.json` | QA pairs per shloka used as few-shot examples in the generator |

---

## Key design decisions

### Why rewrite the query before embedding?
Raw user queries ("I hate my job") are colloquial and short — they embed poorly against philosophical theme descriptions ("Burnout, Career Dissatisfaction, and Purpose Loss"). The rewrite node translates the human problem into richer language, then the classifier uses a combined embedding (average of original + rewritten) so neither signal dominates.

### Why a fixed NON_TEACHING_SHLOKAS blocklist?
Some shlokas are narratively important but have no actionable teaching — Arjuna's laments (1.47, 2.1, 2.2), birth/death philosophical statements (2.26, 2.27, 2.28), and fatalistic verses (18.60). These were consistently retrieved as top matches for distress queries because they embed close to emotional content, but the generator couldn't ground useful advice in them. A blocklist in the retriever is simpler and more reliable than trying to prompt the generator to ignore them.

### Why 70B for the critic, not 8B?
The original 8B critic was demanding near-verbatim shloka phrasing in responses and failing every query on attempt 1, burning 2x API quota with no improvement on retry. 70B understands that paraphrasing the shloka's meaning counts as grounded. After the switch, 5/5 test queries passed on attempt 1.

### Why cap retries at 1 (retry_count >= 2)?
The critic loop exists to catch genuinely bad responses, not to polish good ones. With 70B, most responses pass on attempt 1. Capping at 1 retry means worst case is 2 generator calls + 2 critic calls per query — acceptable latency for a counselling use case.

### Why a hard crisis intercept outside the graph?
The LangGraph pipeline is optimized to give Gita-grounded responses. For suicidal ideation, that's the wrong output regardless of how good the shlokas are. The intercept fires before any API call is made, returns a warm acknowledgement + iCall India number (📞 9152987821), and exits. No Groq or Gemini quota consumed.

### Parallelization
`classifier_node` fires two Gemini embedding calls concurrently (original query + rewritten query) via `ThreadPoolExecutor`, since neither depends on the other. Roughly halves that node's latency. All other nodes are sequential by dependency.

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
```

Run the app:
```bash
streamlit run app.py
```

---

## Stack

- **LangGraph** — pipeline orchestration and state management
- **Groq** — fast inference (Llama 3.3 70B for generation/critic, Llama 3.1 8B for intent)
- **Google Gemini** — embeddings (gemini-embedding-001)
- **Streamlit** — UI
- **scikit-learn** — cosine similarity for retrieval reranking

---

## Known limitations

- **Mixed intent queries** ("explain 2.47 but I feel like I never live up to it") route to lookup only — the emotional half is dropped. Single-intent routing is a known tradeoff.
- **Gemini free-tier rate limits** apply to live embedding calls in `classifier_node`. Under heavy concurrent load this will throttle.
- **18.66 misapplication risk** — "abandon all duties, take refuge in Me" can be misread as "give up" when retrieved for despair queries. Generator prompt includes a note on correct interpretation but this needs monitoring.
- **No persistent memory** — conversation memory is session-scoped in Streamlit. Restarting the app clears all context.