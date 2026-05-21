import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Paper Intel",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Paper Intel")
    st.caption("Agentic RAG over ML/AI papers")
    st.divider()

    # Index status
    try:
        s = requests.get(f"{API_URL}/status", timeout=5).json()
        col1, col2 = st.columns(2)
        col1.metric("Papers", s["indexed_papers"])
        col2.metric("Chunks", s["total_chunks"])
        if s["paper_ids"]:
            with st.expander("Indexed papers"):
                for pid in s["paper_ids"]:
                    st.caption(pid)
    except Exception:
        st.warning("API not reachable")

    st.divider()

    # Ingest single paper
    st.subheader("Ingest a paper")
    arxiv_id = st.text_input("arXiv ID", placeholder="e.g. 2310.06825")
    if st.button("Ingest", use_container_width=True) and arxiv_id.strip():
        with st.spinner(f"Ingesting {arxiv_id}..."):
            resp = requests.post(
                f"{API_URL}/ingest",
                json={"arxiv_ids": [arxiv_id.strip()]},
                timeout=300,
            )
        for r in resp.json():
            if r["status"].startswith("ok"):
                st.success(f"✓ {r['paper_id']} — {r['chunks']} chunks")
            elif "skipped" in r["status"]:
                st.info(f"Already indexed: {r['paper_id']}")
            else:
                st.error(f"✗ {r['paper_id']}: {r['status']}")

    st.divider()

    # Bulk ingest by topic
    st.subheader("Bulk ingest by topic")
    query = st.text_input("Search query", placeholder="e.g. large language model")
    max_p = st.slider("Max papers", 10, 200, 50)
    if st.button("Start bulk ingest", use_container_width=True) and query.strip():
        resp = requests.post(
            f"{API_URL}/ingest/search",
            json={"query": query.strip(), "max_papers": max_p},
            timeout=10,
        )
        if resp.status_code == 200:
            st.success("Bulk ingestion started in background.")
            st.caption("Check progress below.")
        elif resp.status_code == 409:
            st.warning("Already running — wait for it to finish.")
        else:
            st.error(resp.text)

    prog = requests.get(f"{API_URL}/ingest/progress", timeout=5).json()
    if prog["running"]:
        pct = prog["done"] / prog["total"] if prog["total"] else 0
        st.progress(pct, text=f"{prog['done']}/{prog['total']} papers — {prog['query']}")
    elif prog["done"] > 0:
        st.caption(f"Last run: {prog['done']} papers, {prog['skipped']} skipped, {len(prog['errors'])} errors")

# ── main area ─────────────────────────────────────────────────────────────────
st.header("Ask a question")

question = st.text_area(
    "Question",
    placeholder="e.g. How do attention mechanisms work in transformers?",
    height=80,
)

col_a, col_b = st.columns([3, 1])
with col_a:
    max_steps = st.slider("Max agent steps", 2, 10, 6)
with col_b:
    eval_hall = st.checkbox("Hallucination eval", value=False)
    st.caption("Slower — runs fact-checking")

ask_btn = st.button("Ask ▶", type="primary", use_container_width=True)

if ask_btn and question.strip():
    with st.spinner("Agent thinking..."):
        resp = requests.post(
            f"{API_URL}/ask",
            json={
                "question": question.strip(),
                "max_steps": max_steps,
                "eval_hallucination": eval_hall,
            },
            timeout=120,
        )

    if resp.status_code == 200:
        data = resp.json()

        # Answer
        st.subheader("Answer")
        st.markdown(data["answer"])

        # Hallucination report
        if data.get("hallucination"):
            h = data["hallucination"]
            color = {"PASS": "green", "WARN": "orange", "FAIL": "red"}.get(h["verdict"], "gray")
            st.markdown(
                f"**Hallucination check:** :{color}[{h['verdict']}] — "
                f"{h['supported_facts']}/{h['total_facts']} facts grounded "
                f"({h['support_ratio']*100:.0f}%)"
            )
            if h["unsupported"]:
                with st.expander("Unsupported facts"):
                    for f in h["unsupported"]:
                        st.caption(f"✗ {f}")

        st.divider()

        # ReAct trace
        with st.expander(f"🔍 ReAct trace — {data['iterations']} steps"):
            for step in data["trace"]:
                icon = {"hybrid_search": "🔍", "expand_citations": "🕸",
                        "check_contradiction": "⚖", "finalize_answer": "✓"}.get(step["tool"], "→")
                st.markdown(f"**Step {step['iteration']} {icon} `{step['tool']}`**")
                st.code(f"IN:  {step['input_summary'][:120]}\nOUT: {step['result_summary'][:120]}")

        # Sources
        if data["cited_chunks"]:
            with st.expander(f"📚 Sources — {len(data['cited_chunks'])} chunks"):
                for chunk in data["cited_chunks"]:
                    st.markdown(f"**{chunk['paper_title']}** `{chunk['paper_id']}`")
                    st.caption(f"Section: {chunk['section']}")
                    st.text(chunk["text"][:300])
                    st.divider()

    elif resp.status_code == 429:
        st.error(f"Rate limit: {resp.json().get('detail', '')}")
    elif resp.status_code == 400:
        st.warning(resp.json().get("detail", "No papers indexed yet."))
    else:
        st.error(f"Error {resp.status_code}: {resp.text[:200]}")
