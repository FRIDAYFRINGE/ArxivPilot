from __future__ import annotations
import pickle
from collections import deque
from pathlib import Path
from typing import Literal

import networkx as nx

from paper_intel.models.paper import PaperMetadata


class CitationGraph:
    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()

    def add_paper(self, metadata: PaperMetadata) -> None:
        self.graph.add_node(
            metadata.paper_id,
            title=metadata.title,
            year=metadata.year,
            authors=metadata.authors,
        )
        for ref_id in metadata.references:
            self.graph.add_edge(metadata.paper_id, ref_id)

    def get_neighbors(
        self,
        paper_id: str,
        direction: Literal["cites", "cited_by", "both"] = "both",
        depth: int = 1,
    ) -> list[str]:
        if paper_id not in self.graph:
            return []

        visited: set[str] = {paper_id}
        queue: deque[tuple[str, int]] = deque([(paper_id, 0)])
        result: list[str] = []

        while queue:
            node, d = queue.popleft()
            if d >= depth:
                continue
            if direction in ("cites", "both"):
                for neighbor in self.graph.successors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        result.append(neighbor)
                        queue.append((neighbor, d + 1))
            if direction in ("cited_by", "both"):
                for neighbor in self.graph.predecessors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        result.append(neighbor)
                        queue.append((neighbor, d + 1))

        return result

    def find_co_cited(self, paper_id_a: str, paper_id_b: str) -> list[str]:
        """Papers cited by both A and B — shared references indicate same topic."""
        refs_a = set(self.graph.successors(paper_id_a)) if paper_id_a in self.graph else set()
        refs_b = set(self.graph.successors(paper_id_b)) if paper_id_b in self.graph else set()
        return list(refs_a & refs_b)

    def indexed_neighbors(
        self,
        paper_id: str,
        indexed_ids: set[str],
        direction: Literal["cites", "cited_by", "both"] = "both",
        depth: int = 1,
    ) -> list[str]:
        """Like get_neighbors but filters to only papers that are indexed."""
        return [pid for pid in self.get_neighbors(paper_id, direction, depth) if pid in indexed_ids]

    def has_paper(self, paper_id: str) -> bool:
        return paper_id in self.graph

    def paper_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self.graph = pickle.load(f)
