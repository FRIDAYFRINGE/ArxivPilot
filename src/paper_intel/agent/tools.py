TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": "Search indexed papers with hybrid BM25+vector retrieval. Returns relevant chunks with citations. Use first for any factual question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Specific technical search query."},
                    "top_k": {"type": "integer", "description": "Chunks to return (default 2).", "default": 2},
                    "paper_ids": {"type": "array", "items": {"type": "string"}, "description": "Restrict to these arxiv IDs."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_citations",
            "description": "Find papers cited by or citing a given paper, then retrieve matching chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "ArXiv ID to expand from."},
                    "query": {"type": "string", "description": "What to search for in neighbor papers."},
                    "direction": {"type": "string", "enum": ["cites", "cited_by", "both"], "default": "both"},
                    "depth": {"type": "integer", "default": 1},
                },
                "required": ["paper_id", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_contradiction",
            "description": "Compare two claims from different papers: returns AGREE, DISAGREE, or UNCERTAIN.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_a": {"type": "string"},
                    "source_a": {"type": "string", "description": "ArXiv ID of claim A."},
                    "claim_b": {"type": "string"},
                    "source_b": {"type": "string", "description": "ArXiv ID of claim B."},
                },
                "required": ["claim_a", "source_a", "claim_b", "source_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_answer",
            "description": "Produce the final answer. Call once after retrieving evidence. Ends the loop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "Complete answer in plain text, no LaTeX."},
                    "cited_chunk_ids": {"type": "array", "items": {"type": "string"}},
                    "contradiction_flags": {"type": "array", "items": {"type": "string"}, "default": []},
                },
                "required": ["answer", "cited_chunk_ids"],
            },
        },
    },
]
