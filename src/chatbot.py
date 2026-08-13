"""
High-level orchestration: process an uploaded PDF (extract, chunk, embed,
store, summarize) and run multi-turn chat turns against it.

This is the module app.py (Streamlit) and api.py (FastAPI) both call —
neither UI layer talks to chunking/vector_store/graph directly.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.chunking import chunk_document
from src.document_loader import DocumentLoadError, load_pdf
from src.graph import build_graph
from src.llm import get_llm
from src.summarizer import summarize_document
from src.vector_store import VectorStore


@dataclass
class DocumentSession:
    document_id: str
    filename: str
    document_hash: str
    summary: str
    chat_history: list[dict] = field(default_factory=list)


class Assistant:
    """Owns one active document session at a time (per spec section 20)."""

    def __init__(self):
        self._vector_store: VectorStore | None = None
        self._llm = None
        self.session: DocumentSession | None = None
        self._known_hashes: dict[str, str] = {}  # hash -> document_id

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def process_document(self, file_bytes: bytes, filename: str) -> DocumentSession:
        """Loads, chunks, embeds, stores, and summarizes a new PDF.

        Replaces any previously active document session (spec section 20/21).
        Skips re-embedding if this exact file (by hash) was already processed
        (spec section 31).
        """
        doc = load_pdf(file_bytes, filename)  # raises DocumentLoadError

        if doc.document_hash in self._known_hashes:
            document_id = self._known_hashes[doc.document_hash]
            reused = True
        else:
            document_id = str(uuid.uuid4())
            reused = False

        chunks = chunk_document(doc, document_id)

        if not reused:
            self.vector_store.clear_document(document_id)
            self.vector_store.add_chunks(chunks)
            self._known_hashes[doc.document_hash] = document_id

        summary = summarize_document(self.llm, doc.full_text, chunks)

        self.session = DocumentSession(
            document_id=document_id,
            filename=filename,
            document_hash=doc.document_hash,
            summary=summary,
            chat_history=[],
        )
        return self.session

    def ask(self, question: str) -> dict:
        """Runs one multi-turn chat turn. Raises RuntimeError if no document loaded."""
        if self.session is None:
            raise RuntimeError("No document has been processed yet. Upload a PDF first.")

        graph = build_graph(self.llm, self.vector_store)
        result = graph.invoke(
            {
                "question": question,
                "standalone_question": "",
                "chat_history": self.session.chat_history,
                "document_id": self.session.document_id,
                "retrieved_documents": [],
                "answer": "",
                "sources": [],
                "answerable": False,
            }
        )
        if result.get("answerable", False):
            self.session.chat_history.append({"role": "user", "content": question})
            self.session.chat_history.append({"role": "assistant", "content": result["answer"]})

        return {"answer": result["answer"], "sources": result["sources"],"retrieved_documents": result.get("retrieved_documents", []),"standalone_question": result.get("standalone_question",question),}
