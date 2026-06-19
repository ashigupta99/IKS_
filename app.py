import streamlit as st
from gita_graph import run

# -------------------
# Page config
# -------------------

st.set_page_config(
    page_title="Gita Counsellor",
    page_icon="🪷",
    layout="centered"
)

# -------------------
# Session state init
# -------------------

if "memory" not in st.session_state:
    st.session_state.memory = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {role, content, meta}

# -------------------
# Header
# -------------------

st.title("🪷 Bhagavad Gita Assistant")
st.caption("Ask for guidance, look up a shloka by number, or search for verses by topic.")

st.divider()

# -------------------
# Chat history display
# -------------------

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show shloka cards only for assistant messages that have them
        if msg["role"] == "assistant" and msg.get("shlokas_used"):
            intent = msg.get("intent", "")
            with st.expander(
                "📖 Shlokas referenced" if intent != "shloka_search" else "📖 Matching shlokas",
                expanded=False
            ):
                for item in msg["shlokas_used"]:
                    shloka = item["shloka"]
                    sid = shloka.get("id", "")
                    translation = shloka.get("translation_en", "").strip()
                    bhashya = shloka.get("bhashya_hindi", "").strip()

                    st.markdown(f"**[{sid}]**")
                    st.markdown(f"*{translation}*")
                    if bhashya and intent != "shloka_search":
                        st.caption(f"Shankara Bhashya: {bhashya[:200]}{'...' if len(bhashya) > 200 else ''}")
                    st.divider()

        # Show retry info if it happened (counselling only)
        if msg["role"] == "assistant" and msg.get("retry_count", 0) > 1:
            st.caption(f"ℹ️ Response refined after {msg['retry_count']} attempts")

# -------------------
# Input
# -------------------

query = st.chat_input("Share what's on your mind, or ask about a shloka...")

if query:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    # Run the graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run(query, memory=st.session_state.memory)

        response = result["response"]
        intent = result.get("intent", "counselling")
        shlokas_used = result.get("shlokas_used", [])
        retry_count = result.get("retry_count", 0)

        st.markdown(response)

        # Shloka cards
        if shlokas_used:
            with st.expander(
                "📖 Shlokas referenced" if intent != "shloka_search" else "📖 Matching shlokas",
                expanded=False
            ):
                for item in shlokas_used:
                    shloka = item["shloka"]
                    sid = shloka.get("id", "")
                    translation = shloka.get("translation_en", "").strip()
                    bhashya = shloka.get("bhashya_hindi", "").strip()

                    st.markdown(f"**[{sid}]**")
                    st.markdown(f"*{translation}*")
                    if bhashya and intent != "shloka_search":
                        st.caption(f"Shankara Bhashya: {bhashya[:200]}{'...' if len(bhashya) > 200 else ''}")
                    st.divider()

        if retry_count > 1:
            st.caption(f"ℹ️ Response refined after {retry_count} attempts")

    # Save to chat history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "intent": intent,
        "shlokas_used": shlokas_used,
        "retry_count": retry_count
    })

    # Update memory (counselling turns only — lookup/search don't need context)
    if intent == "counselling":
        st.session_state.memory.append({
            "query": query,
            "response": response
        })
        # Keep last 5 turns in memory max
        st.session_state.memory = st.session_state.memory[-5:]

# -------------------
# Sidebar
# -------------------

with st.sidebar:
    st.header("About")
    st.markdown("""
This assistant uses the **Bhagavad Gita** to offer guidance, explain verses, and find relevant shlokas.

**You can ask:**
- 🧘 *Personal problems* — "I feel stuck and unmotivated"
- 🔢 *Specific verses* — "Explain 2.47" or "Chapter 2 shlok 47"
- 🔍 *Topic search* — "Which shloka talks about anger?"
    """)

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.memory = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("Powered by Groq (Llama 3) + Gemini Embeddings + LangGraph")