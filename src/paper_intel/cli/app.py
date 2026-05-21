from __future__ import annotations
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="paper-intel", help="Research Paper Intelligence System", add_completion=False)


def _get_settings():
    from paper_intel.config import settings
    return settings


def _build_retriever(cfg):
    from paper_intel.index.bm25_index import BM25Index
    from paper_intel.index.embedder import Embedder
    from paper_intel.index.vector_store import VectorStore
    from paper_intel.retrieval.dense_retriever import DenseRetriever
    from paper_intel.retrieval.hybrid import HybridRetriever
    from paper_intel.retrieval.sparse_retriever import SparseRetriever

    embedder = Embedder(cfg.embedding_model)
    store = VectorStore(cfg.index_dir, cfg.embedding_dim, cfg.qdrant_url)
    bm25 = BM25Index()
    bm25_path = cfg.index_dir / "bm25.pkl"
    if bm25_path.exists():
        bm25.load(bm25_path)

    dense = DenseRetriever(embedder, store)
    sparse = SparseRetriever(bm25)
    return HybridRetriever(dense, sparse, store, rrf_k=cfg.rrf_k), embedder, bm25, store


def _build_graph(cfg):
    from paper_intel.graph.citation_graph import CitationGraph
    graph = CitationGraph()
    graph_path = cfg.index_dir / "citation_graph.pkl"
    if graph_path.exists():
        graph.load(graph_path)
    return graph


@app.command()
def ingest(
    arxiv_ids: Optional[list[str]] = typer.Argument(None, help="ArXiv paper IDs to ingest"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search arxiv query"),
    max_papers: int = typer.Option(10, "--max", "-n", help="Max papers from search"),
):
    """Download, parse, chunk, and index arxiv papers."""
    from paper_intel.cli.display import console, ingest_progress
    from paper_intel.graph.citation_graph import CitationGraph
    from paper_intel.index.bm25_index import BM25Index
    from paper_intel.index.embedder import Embedder
    from paper_intel.index.vector_store import VectorStore
    from paper_intel.ingestion.arxiv_client import download_pdf, fetch_paper_metadata, search_papers
    from paper_intel.ingestion.chunker import chunk_paper
    from paper_intel.ingestion.pdf_parser import extract_references, parse_pdf

    cfg = _get_settings()

    # Gather papers to ingest
    papers_to_ingest = []
    if search:
        console.print(f"[cyan]Searching arxiv:[/cyan] {search}")
        papers_to_ingest = search_papers(search, max_results=max_papers)
        console.print(f"Found {len(papers_to_ingest)} papers")
    elif arxiv_ids:
        for aid in arxiv_ids:
            console.print(f"[cyan]Fetching:[/cyan] {aid}")
            papers_to_ingest.append(fetch_paper_metadata(aid))
    else:
        console.print("[red]Provide arxiv IDs or --search query[/red]")
        raise typer.Exit(1)

    embedder = Embedder(cfg.embedding_model)
    store = VectorStore(cfg.index_dir, cfg.embedding_dim, cfg.qdrant_url)
    bm25 = BM25Index()
    bm25_path = cfg.index_dir / "bm25.pkl"
    if bm25_path.exists():
        bm25.load(bm25_path)

    graph = CitationGraph()
    graph_path = cfg.index_dir / "citation_graph.pkl"
    if graph_path.exists():
        graph.load(graph_path)

    with ingest_progress() as progress:
        task = progress.add_task("Ingesting papers...", total=len(papers_to_ingest))
        for meta in papers_to_ingest:
            progress.update(task, description=f"[cyan]{meta.paper_id}[/cyan] {meta.title[:40]}...")

            try:
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

                progress.advance(task)
                console.print(f"  ✓ {meta.paper_id}: {len(chunks)} chunks")
            except Exception as e:
                console.print(f"  [red]✗ {meta.paper_id}: {e}[/red]")
                progress.advance(task)

    bm25.save(bm25_path)
    graph.save(graph_path)
    console.print(f"\n[green]Done.[/green] Index: {store.count()} total chunks, "
                  f"{graph.paper_count()} papers, {graph.edge_count()} citation edges.")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question about the indexed papers"),
    decompose: bool = typer.Option(True, "--decompose/--no-decompose", help="Enable query decomposition"),
    eval_hallucination: bool = typer.Option(False, "--eval-hallucination", help="Run hallucination eval"),
    max_steps: int = typer.Option(10, "--max-steps", help="Max agent iterations"),
):
    """Ask a question against the indexed corpus."""
    from openai import OpenAI

    from paper_intel.agent.react_agent import ReactAgent
    from paper_intel.agent.tool_executor import ToolExecutor
    from paper_intel.cli.display import (
        console,
        print_agent_step,
        print_answer,
        print_hallucination_report,
    )
    from paper_intel.reasoning.contradiction import ContradictionChecker
    from paper_intel.reasoning.hallucination import HallucinationEvaluator
    from paper_intel.reasoning.query_decomposer import QueryDecomposer

    cfg = _get_settings()
    if not cfg.llm_api_key:
        console.print("[red]Set LLM_API_KEY in your .env file[/red]")
        raise typer.Exit(1)

    client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
    retriever, _, _, _ = _build_retriever(cfg)
    graph = _build_graph(cfg)

    if retriever.store.count() == 0:
        console.print("[yellow]No papers indexed yet. Run: paper-intel ingest <arxiv-id>[/yellow]")
        raise typer.Exit(1)

    contradiction_checker = ContradictionChecker(client, cfg.llm_model)
    executor = ToolExecutor(retriever, graph, contradiction_checker)

    console.print(f"\n[bold]Q:[/bold] {question}\n")

    final_question = question
    if decompose:
        decomposer = QueryDecomposer(client, cfg.llm_model)
        decomp = decomposer.decompose(question)
        if decomp.needs_decomposition and len(decomp.sub_questions) > 1:
            console.print(f"[dim]Decomposed into {len(decomp.sub_questions)} sub-questions[/dim]")
            waves = decomposer.execution_order(decomp.sub_questions)
            sub_answers: list[tuple[str, str]] = []
            for wave in waves:
                for sq in wave:
                    console.print(f"\n[dim]Sub-question:[/dim] {sq.question}")
                    agent = ReactAgent(
                        client, executor, cfg.llm_model, max_steps, print_agent_step
                    )
                    out = agent.run(sq.question)
                    sub_answers.append((sq.question, out.answer))

            final_answer_text = decomposer.aggregate_answers(question, sub_answers)
            from paper_intel.agent.react_agent import AgentOutput
            output = AgentOutput(
                question=question,
                answer=final_answer_text,
                cited_chunks=[],
                contradiction_flags=[],
                iterations=len(sub_answers),
                success=True,
            )
            print_answer(output)
            return

    agent = ReactAgent(client, executor, cfg.llm_model, max_steps, print_agent_step)
    output = agent.run(final_question)
    print_answer(output)

    if eval_hallucination and output.cited_chunks:
        console.print("\n[dim]Running hallucination evaluation...[/dim]")
        evaluator = HallucinationEvaluator(
            client, cfg.llm_model,
            cfg.hallucination_pass_threshold,
            cfg.hallucination_warn_threshold,
        )
        report = evaluator.evaluate(output.answer, output.cited_chunks)
        print_hallucination_report(report)


@app.command()
def status():
    """Show indexed paper and chunk counts."""
    from paper_intel.cli.display import console

    cfg = _get_settings()
    from paper_intel.index.vector_store import VectorStore
    store = VectorStore(cfg.index_dir, cfg.embedding_dim, cfg.qdrant_url)
    paper_ids = store.indexed_paper_ids()

    graph = _build_graph(cfg)

    console.print(f"[bold]Indexed papers:[/bold] {len(paper_ids)}")
    console.print(f"[bold]Total chunks:[/bold]  {store.count()}")
    console.print(f"[bold]Citation edges:[/bold] {graph.edge_count()}")

    if paper_ids:
        console.print("\n[bold]Papers:[/bold]")
        for pid in paper_ids:
            console.print(f"  • {pid}")


@app.command(name="graph-viz")
def graph_viz(
    paper_id: str = typer.Argument(..., help="ArXiv paper ID"),
    depth: int = typer.Option(1, "--depth", "-d", help="BFS depth"),
):
    """Print citation neighborhood for a paper."""
    from paper_intel.cli.display import console

    cfg = _get_settings()
    graph = _build_graph(cfg)

    if not graph.has_paper(paper_id):
        console.print(f"[yellow]Paper {paper_id} not in citation graph.[/yellow]")
        raise typer.Exit(1)

    cites = graph.get_neighbors(paper_id, direction="cites", depth=depth)
    cited_by = graph.get_neighbors(paper_id, direction="cited_by", depth=depth)

    console.print(f"\n[bold]{paper_id}[/bold] citation neighborhood (depth={depth})")
    if cites:
        console.print(f"\n  [cyan]Cites ({len(cites)}):[/cyan]")
        for pid in cites:
            console.print(f"    → {pid}")
    if cited_by:
        console.print(f"\n  [green]Cited by ({len(cited_by)}):[/green]")
        for pid in cited_by:
            console.print(f"    ← {pid}")
    if not cites and not cited_by:
        console.print("  No neighbors found.")


if __name__ == "__main__":
    app()
