# Research Agent

> Agentic RAG system for ML/AI research papers — ingest arxiv papers, ask deep technical questions, and get cited, grounded answers with a full ReAct reasoning trace.

---

## What it does

You throw an arxiv paper ID at it. It downloads the PDF, parses and chunks the text, embeds the chunks with BGE-small, and stores them in Qdrant. When you ask a question, a ReAct agent loops through hybrid BM25 + vector search steps, checks for contradictions across papers, and synthesises a cited answer grounded entirely in the retrieved evidence.

```
┌──────────────────────────────────────────────────────┐
│  Browser → http://localhost:8501  (Streamlit chat UI) │
└──────────────────────┬───────────────────────────────┘
                       │ HTTP
┌──────────────────────▼───────────────────────────────┐
│  FastAPI server → http://localhost:8000               │
│  • Session-aware multi-turn chat                      │
│  • ReAct agent loop (hybrid retrieval + LLM)          │
│  • Hallucination evaluator                            │
│  • Bulk background ingest                             │
└──────┬──────────────────────────┬────────────────────┘
       │                          │
┌──────▼──────┐          ┌────────▼────────────────────┐
│   Qdrant    │          │  OpenRouter                  │
│  :6333      │          │  (DeepSeek V4 Flash or any   │
│  HNSW index │          │   OpenAI-compatible model)   │
└─────────────┘          └─────────────────────────────┘
```

---

## Features

- **Hybrid retrieval** — BM25 keyword search + dense vector search fused via Reciprocal Rank Fusion
- **ReAct agent** — model reasons step-by-step, decides which papers to search, stops when it has enough evidence
- **Multi-turn chat** — session-based history, ask follow-up questions naturally
- **Citation graph** — traverse references between papers with `expand_citations`
- **Contradiction detection** — automatically flags conflicting claims across different papers
- **Hallucination evaluation** — fact-check answers against retrieved evidence (optional, per-query)
- **Streamlit UI** — full chat interface with ReAct trace + source viewer
- **REST API** — FastAPI backend with auto-generated Swagger docs
- **Bulk ingest** — ingest 100+ papers by topic with a background job and progress endpoint
- **Multi-model** — swap to any OpenAI-compatible provider by changing two env vars

---

## Quick start

### Prerequisites

- Docker Desktop running
- An [OpenRouter](https://openrouter.ai) API key (free tier available)

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd <repo-folder>
cp .env.example .env
```

Edit `.env`:

```env
LLM_API_KEY=sk-or-v1-your-key-here
LLM_MODEL=deepseek/deepseek-v4-flash
LLM_BASE_URL=https://openrouter.ai/api/v1
```

### 2. Start everything

```bash
docker compose up -d
```

Three containers start:

| Container | What it does |
|---|---|
| `research-agent-qdrant-1` | Vector database |
| `research-agent-api-1` | FastAPI + embedding model (BGE-small loads once, stays warm) |
| `research-agent-ui-1` | Streamlit chat UI |

### 3. Ingest papers

```bash
# By arxiv ID
docker compose --profile cli run --rm app ingest 1706.03762 1810.04805 2005.14165 1512.03385
```

Or via the API:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"arxiv_ids": ["1706.03762", "1810.04805"]}'
```

Or bulk-ingest an entire topic in the background:

```bash
curl -X POST http://localhost:8000/ingest/search \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanism", "max_papers": 50}'

# Poll progress
curl http://localhost:8000/ingest/progress
```

### 4. Ask questions

Open `http://localhost:8501` and start chatting, or use the API:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does self-attention work?"}'
```

---

## API reference

### `POST /ask`

```json
{
  "question": "How does BERT differ from GPT-3?",
  "max_steps": 8,
  "eval_hallucination": false,
  "session_id": null
}
```

Pass `session_id` from a previous response to continue a conversation. The server keeps up to 10 turns per session.

**Response:**

```json
{
  "question": "...",
  "answer": "BERT uses masked LM while GPT-3 uses autoregressive LM [Devlin et al., 2018]...",
  "iterations": 4,
  "session_id": "abc-123-...",
  "trace": [
    {"iteration": 1, "tool": "hybrid_search", "input_summary": "...", "result_summary": "..."},
    {"iteration": 2, "tool": "finalize_answer", "input_summary": "...", "result_summary": "..."}
  ],
  "cited_chunks": [
    {"chunk_id": "...", "paper_id": "1810.04805", "paper_title": "BERT...", "section": "...", "text": "..."}
  ],
  "contradiction_flags": [],
  "hallucination": null
}
```

### `POST /ingest`

Ingest papers by arxiv ID. Already-indexed papers are skipped automatically.

```json
{"arxiv_ids": ["1706.03762", "1810.04805"]}
```

### `POST /ingest/search`

Bulk ingest by topic. Returns immediately, runs in background.

```json
{"query": "large language model", "max_papers": 100}
```

### `GET /ingest/progress`

Poll bulk ingest status — `done`, `total`, `running`, `errors`.

### `GET /status`

Index stats: paper count, chunk count, citation edges, paper IDs.

### `DELETE /session/{session_id}`

Clear conversation history for a session.

### `GET /docs`

Interactive Swagger UI — try every endpoint in the browser.

---

## Multi-turn chat

```python
import requests

API = "http://localhost:8000"

# Turn 1
r1 = requests.post(f"{API}/ask", json={"question": "What is self-attention?"})
sid = r1.json()["session_id"]

# Turn 2 — "that" refers to self-attention from Turn 1
r2 = requests.post(f"{API}/ask", json={
    "question": "How does that compare to cross-attention?",
    "session_id": sid
})

# Turn 3
r3 = requests.post(f"{API}/ask", json={
    "question": "Which one does BERT use in its encoder?",
    "session_id": sid
})
```

---

## Configuration

All settings read from `.env`. Restart (no rebuild) picks up changes.

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | — | API key (required) |
| `LLM_MODEL` | `deepseek/deepseek-v4-flash` | Any OpenRouter or OpenAI model ID |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Provider base URL |
| `QDRANT_URL` | _(empty)_ | Empty = local embedded; `http://qdrant:6333` in Docker |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-transformer model |
| `FINAL_TOP_K` | `4` | Chunks returned per search |
| `CHUNK_SIZE_TOKENS` | `350` | Target tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `70` | Overlap between adjacent chunks |

### Switching models

```env
# Free-tier options on OpenRouter
LLM_MODEL=deepseek/deepseek-v4-flash
LLM_MODEL=deepseek/deepseek-r1

# OpenAI direct
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...

# Local Ollama (no API key needed)
LLM_MODEL=llama3.2
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_API_KEY=ollama
```

Restart after changing: `docker compose up -d --force-recreate api`

---

## Docker reference

```bash
# Start all services
docker compose up -d

# View running containers
docker compose ps

# Stream API logs
docker compose logs api -f

# Restart API after .env changes (no rebuild needed)
docker compose up -d --force-recreate api

# Rebuild after code changes
docker compose build api ui
docker compose up -d --force-recreate api ui

# Stop everything (keep data)
docker compose down

# Stop everything and delete all volumes / data
docker compose down -v

# Ingest papers via CLI
docker compose --profile cli run --rm app ingest 1706.03762

# Ask a question via CLI
docker compose --profile cli run --rm app ask "What is self-attention?" --no-decompose

# Copy indexed PDFs to local folder (Windows)
docker run --rm \
  -v research-agent_papers_data:/data \
  -v C:/papers_export:/out \
  alpine sh -c "cp /data/*.pdf /out/"
```

---

## Project structure

```
.
├── src/paper_intel/
│   ├── agent/
│   │   ├── react_agent.py       ReAct loop — search cap, citation fix, session history
│   │   ├── tool_executor.py     Dispatches hybrid_search / expand_citations / check_contradiction
│   │   └── tools.py             OpenAI-format tool schemas
│   ├── ingestion/
│   │   ├── arxiv_client.py      Metadata + PDF download, exponential backoff on 429
│   │   ├── pdf_parser.py        PyMuPDF extraction, NFKC normalisation, section detection
│   │   └── chunker.py           Sentence-aware 350-token sliding window
│   ├── index/
│   │   ├── embedder.py          BGE-small wrapper (CPU)
│   │   ├── vector_store.py      Qdrant upsert / search / scroll
│   │   └── bm25_index.py        In-memory BM25Okapi, incremental rebuild
│   ├── retrieval/
│   │   ├── dense_retriever.py   Query embedding → Qdrant ANN search
│   │   ├── sparse_retriever.py  BM25 keyword search
│   │   └── hybrid.py            Reciprocal Rank Fusion of dense + sparse
│   ├── graph/
│   │   └── citation_graph.py    NetworkX directed citation graph
│   ├── reasoning/
│   │   ├── contradiction.py     Pairwise claim comparison via LLM
│   │   ├── hallucination.py     Atomic fact extraction + parallel verification
│   │   └── query_decomposer.py  Multi-hop question decomposition
│   ├── server/
│   │   └── app.py               FastAPI — session management, bulk ingest, error handling
│   ├── ui/
│   │   └── streamlit_app.py     Chat UI with trace, sources, hallucination display
│   ├── cli/
│   │   └── app.py               Typer CLI (ingest / ask / status / graph-viz)
│   └── config.py                pydantic-settings, reads from .env
├── tests/
├── Dockerfile                   Multi-stage: builder (gcc) → slim runtime (no gcc)
├── Dockerfile.ui                Lightweight Streamlit-only image (~300 MB vs 3 GB)
├── docker-compose.yml           qdrant + api + ui services, cli profile
├── .env.example
└── pyproject.toml
```

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit 1.35+ |
| API | FastAPI + Uvicorn |
| LLM | DeepSeek V4 Flash via OpenRouter (OpenAI SDK) |
| Embeddings | BGE-small-en-v1.5 — sentence-transformers, CPU |
| Vector DB | Qdrant 1.18 — HNSW indexing, cosine similarity |
| Sparse index | BM25Okapi — rank-bm25, in-memory |
| PDF parsing | PyMuPDF |
| Citation graph | NetworkX |
| Config | pydantic-settings |
| Runtime | Python 3.11 · Docker Compose |

---

## Known limitations

| Issue | Detail |
|---|---|
| Table retrieval | Hyperparameter tables split across chunk boundaries may retrieve from the wrong paper |
| LLM credit exhaustion | OpenRouter free tier has daily/monthly caps — the API returns a clear `402` when credits run out |
| BM25 rebuild cost | BM25Okapi rebuilds from scratch on every add — at 100k+ chunks switch to Elasticsearch or Qdrant sparse vectors |
| Single collection | All papers share one Qdrant collection — no per-user or per-domain isolation |

---

## Running tests

```bash
docker compose --profile cli run --rm app python -m pytest tests/ -v
```
