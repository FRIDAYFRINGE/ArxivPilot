from __future__ import annotations
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from paper_intel.agent.react_agent import AgentOutput
    from paper_intel.reasoning.hallucination import HallucinationReport

console = Console()


def print_agent_step(tool_name: str, input_summary: str, result_summary: str) -> None:
    icon = {
        "hybrid_search": "🔍",
        "expand_citations": "🕸",
        "check_contradiction": "⚖",
        "finalize_answer": "✓",
    }.get(tool_name, "→")
    console.print(f"  [dim]{icon} {tool_name}[/dim]  [italic]{input_summary[:70]}[/italic]")


def print_answer(output: "AgentOutput") -> None:
    console.print()
    console.print(Panel(
        output.answer,
        title=f"[bold green]Answer[/bold green] ({output.iterations} agent steps)",
        border_style="green",
        padding=(1, 2),
    ))

    if output.contradiction_flags:
        flags_text = "\n".join(f"• {f}" for f in output.contradiction_flags)
        console.print(Panel(
            flags_text,
            title="[bold red]Contradictions Detected[/bold red]",
            border_style="red",
        ))

    if output.cited_chunks:
        table = Table(title="Sources", show_header=True, header_style="bold cyan")
        table.add_column("Paper", style="dim", width=40)
        table.add_column("Section", width=25)
        table.add_column("Authors", width=25)
        table.add_column("Year", width=6)
        seen = set()
        for chunk in output.cited_chunks:
            key = chunk.paper_id
            if key not in seen:
                seen.add(key)
                table.add_row(
                    chunk.paper_title[:38],
                    chunk.section[:23],
                    ", ".join(chunk.authors)[:23],
                    str(chunk.year),
                )
        console.print(table)


def print_hallucination_report(report: "HallucinationReport") -> None:
    color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[report.verdict]
    ratio_pct = f"{report.support_ratio * 100:.0f}%"

    console.print(Panel(
        f"[bold]Verdict: [{color}]{report.verdict}[/{color}][/bold]  "
        f"Support ratio: [{color}]{ratio_pct}[/{color}]  "
        f"({sum(1 for f in report.atomic_facts if f.supported)}/{len(report.atomic_facts)} facts grounded)",
        title="[bold]Hallucination Evaluation[/bold]",
        border_style=color,
    ))

    if report.hallucinated_facts:
        console.print("[red]Unsupported facts:[/red]")
        for af in report.hallucinated_facts:
            console.print(f"  [red]✗[/red] {af.fact_text}")
            if af.rationale:
                console.print(f"    [dim]{af.rationale}[/dim]")


def ingest_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )
