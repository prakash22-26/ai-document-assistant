"""Thin wrapper around VectorStore.similarity_search for the RAG graph."""
from __future__ import annotations

from src.vector_store import VectorStore


def retrieve(vector_store: VectorStore, query: str, document_id: str, top_k: int | None = None) -> list[dict]:
    """Returns top_k relevant chunks for a query, scoped to one document."""
    return vector_store.similarity_search(query=query, document_id=document_id, top_k=top_k)
