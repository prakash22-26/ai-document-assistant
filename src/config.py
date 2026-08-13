"""
Centralized configuration for the AI Document Assistant.

All tunables (LLM provider, chunking, retrieval, embeddings) live here and
are sourced from environment variables / .env so nothing is hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    hf_api_key: str = field(default_factory=lambda: os.getenv("HF_API_KEY", ""))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", ""))

    # Embeddings
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
    )

    # Chunking
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 150))

    # Retrieval
    # Retrieval
    top_k: int = field(
        default_factory=lambda: _get_int("TOP_K", 7)
    )
    retrieval_candidates: int = field(
        default_factory=lambda: _get_int(
            "RETRIEVAL_CANDIDATES",
            15
        )
    )

    rerank_model: str = field(
        default_factory=lambda: os.getenv(
            "RERANK_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )

    

    # Storage
    chroma_persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "chroma_db"))
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "data"))

    # Uploads
    max_upload_mb: int = field(default_factory=lambda: _get_int("MAX_UPLOAD_MB", 25))


settings = Settings()
