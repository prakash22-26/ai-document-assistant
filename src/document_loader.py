"""
PDF loading and page-wise text extraction.

Responsible ONLY for turning a PDF file into a list of (page_number, text)
pairs plus basic validation. Chunking happens downstream in chunking.py.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class DocumentLoadError(Exception):
    """Raised when a PDF cannot be loaded or contains no usable text."""


@dataclass
class PageText:
    page: int  # 1-indexed
    text: str


@dataclass
class LoadedDocument:
    filename: str
    document_hash: str
    pages: list[PageText]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def is_empty(self) -> bool:
        return len(self.full_text.strip()) == 0


def compute_document_hash(file_bytes: bytes) -> str:
    """Stable hash used to detect duplicate uploads and skip re-embedding."""
    return hashlib.sha256(file_bytes).hexdigest()


def load_pdf(file_bytes: bytes, filename: str) -> LoadedDocument:
    """
    Extract page-wise text from PDF bytes.

    Raises DocumentLoadError with a user-friendly message on:
    - corrupt / invalid PDF
    - zero-page PDF
    - fully empty (e.g. scanned, image-only) PDF
    """
    if not file_bytes:
        raise DocumentLoadError("The uploaded file is empty.")

    if not filename.lower().endswith(".pdf"):
        raise DocumentLoadError("Only PDF files are supported in this version.")

    max_bytes = 25 * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise DocumentLoadError("File is too large. Maximum supported size is 25 MB.")

    import io

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except PdfReadError as exc:
        raise DocumentLoadError(f"Could not read this PDF — it may be corrupted: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise DocumentLoadError(f"Unexpected error opening PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise DocumentLoadError("This PDF is password-protected and cannot be read.")

    if len(reader.pages) == 0:
        raise DocumentLoadError("This PDF has no pages.")

    pages: list[PageText] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(PageText(page=idx, text=text.strip()))

    doc = LoadedDocument(
        filename=filename,
        document_hash=compute_document_hash(file_bytes),
        pages=pages,
    )

    if doc.is_empty:
        raise DocumentLoadError(
            "No extractable text was found in this PDF. It is likely a scanned / "
            "image-only document. OCR is not supported in Version 1 — please upload "
            "a text-based PDF."
        )

    return doc
