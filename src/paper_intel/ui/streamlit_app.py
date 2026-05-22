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

# ── session init ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "eval_hallucination" not in st.session_state:
    st.session_state.eval_hallucination = False


# ── helper — MUST be defined before any call site ────────────────────────────
def _render_meta(meta: dict) -> None:
    """Render trace, sources and hallucination report under an assistant bubble."""
    cols = st.columns(3)
    cols[0].caption(f"🔁 {meta['iterations']} steps")

    if meta.get("hallucination"):
        h = meta["hallucination"]
        icon = {"PASS": "🟢", "WARN": "🟡", "FAIL": "🔴"}.get(h["verdict"], "⚪")
        cols[1].caption(f"{icon} {h['verdict']} — {h['supported_facts']}/{h['total_facts']} facts")

    with st.expander(f"🔍 ReAct trace ({meta['iterations']} steps)"):
        for step in meta["trace"]:
            icon_map = {"hybrid_search": "🔍", "expand_citations": "🕸",
                        "check_contradiction": "⚖", "finalize_answer": "✓"}
            st.markdown(f"**{icon_map.get(step['tool'], '→')} Step {step['iteration']} `{step['tool']}`**")
            st.code(
                f"IN:  {step['input_summary'][:120]}\nOUT: {step['result_summary'][:120]}",
                language=None,
            )

    if meta.get("cited_chunks"):
        with st.expander(f"📚 Sources ({len(meta['cited_chunks'])} chunks)"):
            for chunk in meta["cited_chunks"]:
                st.markdown(f"**{chunk['paper_title']}** `{chunk['paper_id']}`")
                st.caption(f"Section: {chunk['section']}")
                st.text(chunk["text"][:250])
                st.divider()

    if meta.get("hallucination") and meta["hallucination"].get("unsupported"):
        with st.expander("⚠️ Unsupported facts"):
            for f in meta["hallucination"]["unsupported"]:
                st.caption(f"✗ {f}")


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Paper Intel")
    st.caption("Agentic RAG over ML/AI papers")
    st.divider()

    try:
        s = requests.get(f"{API_URL}/status", timeout=5).json()
        col1, col2 = st.columns(2)
        col1.metric("Papers", s["indexed_papers"])
        col2.metric("Chunks", s["total_chunks"])
        with st.expander("Indexed papers"):
            for pid in s["paper_ids"]:
                st.caption(pid)
    except Exception:
        st.warning("API not reachable")

    st.divider()

    st.subheader("Chat settings")
    st.session_state.eval_hallucination = st.checkbox(
        "Hallucination eval", value=st.session_state.eval_hallucination,
        help="Checks up to 8 facts against retrieved chunks. Adds ~15-30s."
    )
    max_steps = st.slider("Max agent steps", 2, 10, 6)

    if st.button("🗑 New conversation", use_container_width=True):
        if st.session_state.session_id:
            try:
                requests.delete(f"{API_URL}/session/{st.session_state.session_id}", timeout=5)
            except Exception:
                pass
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    st.divider()

    st.subheader("Add papers")
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

    query = st.text_input("Bulk ingest by topic", placeholder="e.g. diffusion models")
    max_p  = st.slider("Max papers", 10, 200, 30)
    if st.button("Start bulk ingest", use_container_width=True) and query.strip():
        resp = requests.post(
            f"{API_URL}/ingest/search",
            json={"query": query.strip(), "max_papers": max_p},
            timeout=10,
        )
        if resp.status_code == 200:
            st.success("Ingestion started in background.")
        elif resp.status_code == 409:
            st.warning("Already running.")
        else:
            st.error(resp.text)

    try:
        prog = requests.get(f"{API_URL}/ingest/progress", timeout=5).json()
        if prog["running"]:
            pct = prog["done"] / prog["total"] if prog["total"] else 0
            st.progress(pct, text=f"{prog['done']}/{prog['total']} — {prog['query'][:30]}")
    except Exception:
        pass


# ── chat area ─────────────────────────────────────────────────────────────────
st.header("💬 Research Assistant")

if st.session_state.session_id:
    st.caption(f"Session `{st.session_state.session_id[:8]}…`  —  {len(st.session_state.messages)//2} turns")
else:
    st.caption("Ask a question to start a new session.")

# Replay all prior messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            _render_meta(msg["meta"])

# ── input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about the indexed papers…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and reasoning…"):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": prompt,
                        "max_steps": max_steps,
                        "eval_hallucination": st.session_state.eval_hallucination,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=300,
                )
            except requests.exceptions.Timeout:
                st.error("Request timed out — try fewer agent steps or disable hallucination eval.")
                st.stop()
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Is the server running?")
                st.stop()

        if resp.status_code == 200:
            data = resp.json()
            st.session_state.session_id = data["session_id"]
            st.markdown(data["answer"])

            meta = {
                "iterations":  data["iterations"],
                "trace":        data["trace"],
                "cited_chunks": data["cited_chunks"],
                "hallucination": data.get("hallucination"),
            }
            _render_meta(meta)

            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "meta": meta,
            })

        elif resp.status_code == 429:
            st.error(f"Rate limit — {resp.json().get('detail', 'try again later')}")
        elif resp.status_code == 400:
            st.warning(resp.json().get("detail", "No papers indexed yet."))
        else:
            st.error(f"Error {resp.status_code}: {resp.text[:200]}")
