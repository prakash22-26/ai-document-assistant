"""
Typed state passed through the LangGraph conversational RAG workflow.

Fields:
    question:            the user's raw question for this turn
    standalone_question:  question rewritten to be independent of history
    chat_history:        list of {"role": "user"|"assistant", "content": str}
    document_id:         which document's chunks to retrieve from
    retrieved_documents: chunks retrieved from ChromaDB for this turn
    answer:              the LLM's grounded answer
    sources:              deduped list of page numbers cited in the answer
"""
from __future__ import annotations

from typing import TypedDict


class ChatTurn(TypedDict):
    role: str  # "user" or "assistant"
    content: str


class RetrievedChunk(TypedDict):
    text: str
    page: int
    source: str
    chunk_id: str
    distance: float


class GraphState(TypedDict):
    question: str
    standalone_question: str
    chat_history: list[ChatTurn]
    document_id: str
    retrieved_documents: list[RetrievedChunk]
    answer: str
    sources: list[int]
    answerable: bool
