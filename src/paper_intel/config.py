from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_api_key: str = ""
    llm_model: str = "deepseek/deepseek-v4-flash"
    llm_base_url: str = "https://openrouter.ai/api/v1"

    qdrant_url: str = ""  # empty = local embedded mode; set to http://qdrant:6333 in Docker

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 70

    dense_candidates: int = 20
    sparse_candidates: int = 20
    final_top_k: int = 8
    rrf_k: int = 60

    data_dir: Path = Path("data")
    papers_dir: Path = Path("data/papers")
    index_dir: Path = Path("data/index")

    max_agent_iterations: int = 10

    hallucination_pass_threshold: float = 0.90
    hallucination_warn_threshold: float = 0.70

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
