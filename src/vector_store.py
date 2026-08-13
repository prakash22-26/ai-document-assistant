"""
ChromaDB persistence layer.

Single-method dense retrieval: embed the query, cosine-similarity search
against Chroma's HNSW index, return the top_k nearest chunks.
"""
from __future__ import annotations

import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings as ChromaSettings
from functools import lru_cache

from sentence_transformers import CrossEncoder
from src.chunking import Chunk
from src.config import settings
from src.embeddings import get_embedding_function

COLLECTION_NAME = "documents"

@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(
        settings.rerank_model
    )
class VectorStore:
    def __init__(self, persist_dir: str | None = None):
        self._persist_dir = persist_dir or settings.chroma_persist_dir
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)
        self._embed_fn = None

    @property
    def _embed(self):
        if self._embed_fn is None:
            self._embed_fn = get_embedding_function()
        return self._embed_fn

    def clear_document(self, document_id: str) -> None:
        existing = self._collection.get(where={"document_id": document_id})
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)

    def has_document(self, document_id: str) -> bool:
        existing = self._collection.get(where={"document_id": document_id}, limit=1)
        return bool(existing.get("ids"))

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = self._embed.embed_documents(texts)
        self._collection.add(
            ids=[f"{c.document_id}::{c.chunk_id}" for c in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {
                    "document_id": c.document_id,
                    "source": c.source,
                    "page": c.page,
                    "chunk_id": c.chunk_id,
                }
                for c in chunks
            ],
        )

    def similarity_search(
        self,
        query: str,
        document_id: str,
        top_k: int | None = None
    ) -> list[dict]:

        top_k = top_k or settings.top_k

        candidate_k = max(
            settings.retrieval_candidates,
            top_k
        )

        query_vector = self._embed.embed_query(
            query
        )

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=candidate_k,
            where={
                "document_id": document_id
            },
        )

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        if not documents[0]:
            return []

        candidates = []

        for text, metadata, distance in zip(
            documents[0],
            metadatas[0],
            distances[0]
        ):
            candidates.append(
                {
                    "text": text,
                    "page": metadata.get("page"),
                    "source": metadata.get("source"),
                    "chunk_id": metadata.get("chunk_id"),
                    "distance": float(distance),
                }
            )

        reranker = get_reranker()

        pairs = [
            [query, candidate["text"]]
            for candidate in candidates
        ]

        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True
        )

        return [
            candidate
            for candidate, _score in ranked[:top_k]
        ]