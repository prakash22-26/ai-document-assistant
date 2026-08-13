"""
Splits page-wise document text into overlapping chunks while preserving
page-number metadata on every chunk.

Strategy: LangChain's RecursiveCharacterTextSplitter, chunk_size ~1000 chars
with ~150 char overlap by default (configurable via .env). Recursive
splitting is used (rather than fixed-size) because it tries paragraph ->
sentence -> word boundaries in order, which keeps chunks semantically
coherent instead of cutting mid-sentence. Overlap keeps context that
would otherwise be lost at a chunk boundary from a retrieval standpoint.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.document_loader import LoadedDocument


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: int
    source: str
    document_id: str


def chunk_document(doc: LoadedDocument, document_id: str) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    counter = 0
    for page in doc.pages:
        if not page.text.strip():
            continue
        for piece in splitter.split_text(page.text):
            if not piece.strip():
                continue
            counter += 1
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{counter:04d}",
                    text=piece.strip(),
                    page=page.page,
                    source=doc.filename,
                    document_id=document_id,
                )
            )

    return chunks
