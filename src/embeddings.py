"""
Embedding model wrapper.

Uses a local Hugging Face sentence-transformers model so no external API
call (or cost) is needed for embeddings — only the LLM call for
summarization/answering requires an API key.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings


@lru_cache(maxsize=1)
def get_embedding_function() -> HuggingFaceEmbeddings:
    """Cached so the model is loaded into memory only once per process."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
        
    )


