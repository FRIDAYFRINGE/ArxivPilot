from __future__ import annotations
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from paper_intel.config import settings as cfg


def _clean_section(section: str) -> str:
    """Strip arxiv headers and other noise that slips through as section names."""
    import re
    if re.search(r"arXiv\s*:|^\d{4}\.\d{4,5}", section, re.IGNORECASE):
        return "Unknown"
    return section.strip() or "Unknown"


# ── state loaded ONCE at startup, reused for every request ──────────────────
_S: dict[str, Any] = {}

# ── bulk ingestion progress (shared across threads) ─────────────────────────
_PROGRESS: dict[str, Any] = {
    "running": False,
    "query": "",
    "done": 0,
    "total": 0,
    "errors": [],
    "skipped": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model, indexes, and LLM client once at startup."""
    from openai import OpenAI
    from paper_intel.agent.tool_executor import ToolExecutor
    from paper_intel.graph.citation_graph import CitationGraph
    from paper_intel.index.bm25_index import BM25Index
    from paper_intel.index.embedder import Embedder
    from paper_intel.index.vector_store import VectorStore
    from paper_intel.reasoning.contradiction import ContradictionChecker
    from paper_intel.retrieval.dense_retriever import DenseRetriever
    from paper_intel.retrieval.hybrid import HybridRetriever
    from paper_intel.retrieval.sparse_retriever import SparseRetriever

    print("Loading embedding model...", flush=True)
    embedder = Embedder(cfg.embedding_model)

    print("Connecting to Qdrant...", flush=True)
    store = VectorStore(cfg.index_dir, cfg.embedding_dim, cfg.qdrant_url)

    bm25 = BM25Index()
    bm25_path = cfg.index_dir / "bm25.pkl"
    if bm25_path.exists():
        bm25.load(bm25_path)

    graph = CitationGraph()
    graph_path = cfg.index_dir / "citation_graph.pkl"
    if graph_path.exists():
        graph.load(graph_path)

    client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
    contradiction_checker = ContradictionChecker(client, cfg.llm_model)
    retriever = HybridRetriever(
        DenseRetriever(embedder, store),
        SparseRetriever(bm25),
        store,
        rrf_k=cfg.rrf_k,
    )
    executor = ToolExecutor(retriever, graph, contradiction_checker)

    _S.update({
        "embedder": embedder,
        "store": store,
        "bm25": bm25,
        "bm25_path": bm25_path,
        "graph": graph,
        "graph_path": graph_path,
        "client": client,
        "executor": executor,
    })
    print("Server ready.", flush=True)
    yield


app = FastAPI(title="Paper Intel API", version="0.1.0", lifespan=lifespan)


# ── request / response models ────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    max_steps: int = 6
    eval_hallucination: bool = False


class ChunkRef(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    section: str
    text: str


class TraceStep(BaseModel):
    iteration: int
    tool: str
    input_summary: str
    result_summary: str


class HallucinationReport(BaseModel):
    verdict: str            # PASS / WARN / FAIL
    support_ratio: float
    supported_facts: int
    total_facts: int
    unsupported: list[str]  # fact texts that had no grounding


class AskResponse(BaseModel):
    question: str
    answer: str
    iterations: int
    trace: list[TraceStep]
    cited_chunks: list[ChunkRef]
    contradiction_flags: list[str]
    hallucination: HallucinationReport | None = None


class IngestRequest(BaseModel):
    arxiv_ids: list[str]


class IngestResult(BaseModel):
    paper_id: str
    chunks: int
    status: str


class StatusResponse(BaseModel):
    indexed_papers: int
    total_chunks: int
    citation_edges: int
    paper_ids: list[str]


# ── endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "model": cfg.llm_model}


@app.get("/status", response_model=StatusResponse)
def status():
    store = _S["store"]
    graph = _S["graph"]
    paper_ids = store.indexed_paper_ids()
    return StatusResponse(
        indexed_papers=len(paper_ids),
        total_chunks=store.count(),
        citation_edges=graph.edge_count(),
        paper_ids=paper_ids,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    from openai import RateLimitError
    from paper_intel.agent.react_agent import ReactAgent

    if _S["store"].count() == 0:
        raise HTTPException(status_code=400, detail="No papers indexed yet. POST /ingest first.")

    agent = ReactAgent(
        _S["client"],
        _S["executor"],
        cfg.llm_model,
        req.max_steps,
    )
    try:
        output = agent.run(req.question)
    except RateLimitError as e:
        import re
        wait = re.search(r"try again in ([\w\d\.]+)", str(e))
        detail = f"Groq rate limit reached. {('Try again in ' + wait.group(1)) if wait else 'Please wait and retry.'}"
        raise HTTPException(status_code=429, detail=detail)
    except Exception as e:
        msg = str(e)
        if "model_permission" in msg or "403" in msg or "PermissionDenied" in type(e).__name__:
            raise HTTPException(
                status_code=403,
                detail=f"Model '{cfg.llm_model}' is blocked or permission denied. "
                       f"Check your API key and model name at https://openrouter.ai/models",
            )
        if "model_decommissioned" in msg or "decommissioned" in msg:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{cfg.llm_model}' is decommissioned. "
                       f"Update LLM_MODEL in your .env — see https://openrouter.ai/models",
            )
        raise

    return AskResponse(
        question=output.question,
        answer=output.answer,
        iterations=output.iterations,
        trace=[
            TraceStep(
                iteration=s.iteration,
                tool=s.tool,
                input_summary=s.input_summary,
                result_summary=s.result_summary,
            )
            for s in output.trace
        ],
        cited_chunks=[
            ChunkRef(
                chunk_id=c.chunk_id,
                paper_id=c.paper_id,
                paper_title=c.paper_title,
                section=_clean_section(c.section),
                text=c.text[:300].strip(),
            )
            for c in output.cited_chunks
        ],
        contradiction_flags=output.contradiction_flags,
        hallucination=_run_hallucination_eval(output, req) if req.eval_hallucination else None,
    )


def _run_hallucination_eval(output, req) -> "HallucinationReport | None":
    from paper_intel.reasoning.hallucination import HallucinationEvaluator
    try:
        evaluator = HallucinationEvaluator(
            _S["client"], cfg.llm_model,
            cfg.hallucination_pass_threshold,
            cfg.hallucination_warn_threshold,
        )
        report = evaluator.evaluate(output.answer, output.cited_chunks)
        supported = sum(1 for f in report.atomic_facts if f.supported)
        return HallucinationReport(
            verdict=report.verdict,
            support_ratio=report.support_ratio,
            supported_facts=supported,
            total_facts=len(report.atomic_facts),
            unsupported=[f.fact_text for f in report.hallucinated_facts],
        )
    except Exception:
        return None


@app.post("/ingest", response_model=list[IngestResult])
def ingest(req: IngestRequest):
    from paper_intel.ingestion.arxiv_client import download_pdf, fetch_paper_metadata
    from paper_intel.ingestion.chunker import chunk_paper
    from paper_intel.ingestion.pdf_parser import extract_references, parse_pdf

    embedder = _S["embedder"]
    store    = _S["store"]
    bm25     = _S["bm25"]
    graph    = _S["graph"]
    results: list[IngestResult] = []

    already_indexed = set(store.indexed_paper_ids())

    for arxiv_id in req.arxiv_ids:
        if arxiv_id in already_indexed:
            results.append(IngestResult(paper_id=arxiv_id, chunks=0, status="skipped (already indexed)"))
            continue
        try:
            meta = fetch_paper_metadata(arxiv_id)
            pdf_path = download_pdf(meta, cfg.papers_dir)
            pages = parse_pdf(pdf_path)
            meta.references = extract_references(pdf_path)
            chunks = chunk_paper(
                pages, meta,
                chunk_size_tokens=cfg.chunk_size_tokens,
                overlap_tokens=cfg.chunk_overlap_tokens,
            )
            embeddings = embedder.embed_texts([c.text for c in chunks])
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i].tolist()

            store.upsert_chunks(chunks)
            bm25.add_chunks(chunks)
            graph.add_paper(meta)
            already_indexed.add(arxiv_id)
            results.append(IngestResult(paper_id=arxiv_id, chunks=len(chunks), status="ok"))
        except Exception as e:
            results.append(IngestResult(paper_id=arxiv_id, chunks=0, status=f"error: {e}"))

    if any(r.status == "ok" for r in results):
        bm25.save(_S["bm25_path"])
        graph.save(_S["graph_path"])

    return results


# ── bulk search-and-ingest endpoints ─────────────────────────────────────────

class BulkIngestRequest(BaseModel):
    query: str
    max_papers: int = 100


class BulkIngestStarted(BaseModel):
    message: str
    query: str
    max_papers: int


class ProgressResponse(BaseModel):
    running: bool
    query: str
    done: int
    total: int
    skipped: int
    errors: list[str]
    indexed_chunks: int


def _process_one_paper(meta, already_indexed: set, lock: threading.Lock) -> str:
    """Download, parse, chunk, embed and store one paper. Thread-safe."""
    from paper_intel.ingestion.arxiv_client import download_pdf
    from paper_intel.ingestion.chunker import chunk_paper
    from paper_intel.ingestion.pdf_parser import extract_references, parse_pdf

    if meta.paper_id in already_indexed:
        return "skipped"

    pdf_path = download_pdf(meta, cfg.papers_dir)
    pages    = parse_pdf(pdf_path)
    meta.references = extract_references(pdf_path)
    chunks   = chunk_paper(
        pages, meta,
        chunk_size_tokens=cfg.chunk_size_tokens,
        overlap_tokens=cfg.chunk_overlap_tokens,
    )
    embeddings = _S["embedder"].embed_texts([c.text for c in chunks])
    for i, chunk in enumerate(chunks):
        chunk.embedding = embeddings[i].tolist()

    # Qdrant and BM25 writes are not thread-safe — serialize them
    with lock:
        _S["store"].upsert_chunks(chunks)
        _S["bm25"].add_chunks(chunks)
        _S["graph"].add_paper(meta)
        already_indexed.add(meta.paper_id)

    return f"ok:{len(chunks)}"


def _bulk_ingest_worker(query: str, max_papers: int) -> None:
    """Runs in a background thread — searches arxiv and ingests each paper."""
    import concurrent.futures
    from paper_intel.ingestion.arxiv_client import search_papers

    _PROGRESS.update({"running": True, "query": query, "done": 0,
                       "total": 0, "errors": [], "skipped": 0})

    try:
        print(f"[bulk] Searching arxiv: {query!r} max={max_papers}", flush=True)
        papers = search_papers(query, max_results=max_papers)
        _PROGRESS["total"] = len(papers)
        print(f"[bulk] Found {len(papers)} papers — starting parallel ingest", flush=True)

        already_indexed = set(_S["store"].indexed_paper_ids())
        lock = threading.Lock()

        # PDF downloads + parsing + embedding run in parallel (up to 4 workers).
        # Qdrant/BM25 writes are serialized via the lock inside _process_one_paper.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_process_one_paper, meta, already_indexed, lock): meta
                for meta in papers
            }
            for future in concurrent.futures.as_completed(futures):
                meta = futures[future]
                try:
                    result = future.result()
                    if result == "skipped":
                        _PROGRESS["skipped"] += 1
                    else:
                        chunks_n = result.split(":")[1]
                        print(f"[bulk] ✓ {meta.paper_id} ({chunks_n} chunks)", flush=True)
                except Exception as e:
                    err = f"{meta.paper_id}: {e}"
                    _PROGRESS["errors"].append(err)
                    print(f"[bulk] ✗ {err}", flush=True)
                finally:
                    _PROGRESS["done"] += 1

        bm25 = _S["bm25"]
        bm25.save(_S["bm25_path"])
        _S["graph"].save(_S["graph_path"])
        print(f"[bulk] Done. {_PROGRESS['done']} processed, "
              f"{_PROGRESS['skipped']} skipped, {len(_PROGRESS['errors'])} errors.", flush=True)
    finally:
        _PROGRESS["running"] = False


@app.post("/ingest/search", response_model=BulkIngestStarted)
def ingest_search(req: BulkIngestRequest, background_tasks: BackgroundTasks):
    """
    Search arxiv by query and ingest all results in the background.
    Returns immediately — poll GET /ingest/progress to track completion.
    """
    if _PROGRESS["running"]:
        raise HTTPException(
            status_code=409,
            detail="A bulk ingestion is already running. "
                   "Check GET /ingest/progress or wait for it to finish.",
        )
    if req.max_papers < 1 or req.max_papers > 1000:
        raise HTTPException(status_code=400, detail="max_papers must be 1–1000.")

    background_tasks.add_task(_bulk_ingest_worker, req.query, req.max_papers)
    return BulkIngestStarted(
        message="Bulk ingestion started in the background.",
        query=req.query,
        max_papers=req.max_papers,
    )


@app.get("/ingest/progress", response_model=ProgressResponse)
def ingest_progress():
    """Check the status of an ongoing or completed bulk ingestion."""
    return ProgressResponse(
        running=_PROGRESS["running"],
        query=_PROGRESS["query"],
        done=_PROGRESS["done"],
        total=_PROGRESS["total"],
        skipped=_PROGRESS["skipped"],
        errors=_PROGRESS["errors"][-20:],  # last 20 errors only
        indexed_chunks=_S["store"].count() if _S else 0,
    )
