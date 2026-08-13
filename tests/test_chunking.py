from src.chunking import chunk_document
from src.document_loader import LoadedDocument, PageText


def make_doc(pages: list[str]) -> LoadedDocument:
    return LoadedDocument(
        filename="test.pdf",
        document_hash="abc123",
        pages=[PageText(page=i + 1, text=t) for i, t in enumerate(pages)],
    )


def test_chunk_document_preserves_page_metadata():
    doc = make_doc(["Page one text. " * 50, "Page two text. " * 50])
    chunks = chunk_document(doc, document_id="doc-1")

    assert len(chunks) > 0
    assert all(c.document_id == "doc-1" for c in chunks)
    assert all(c.source == "test.pdf" for c in chunks)
    pages_seen = {c.page for c in chunks}
    assert pages_seen == {1, 2}


def test_chunk_document_skips_empty_pages():
    doc = make_doc(["Some real content here that is long enough.", "   ", ""])
    chunks = chunk_document(doc, document_id="doc-2")

    assert all(c.text.strip() for c in chunks)
    assert all(c.page == 1 for c in chunks)


def test_chunk_ids_are_unique_and_sequential():
    doc = make_doc(["Content " * 300])
    chunks = chunk_document(doc, document_id="doc-3")

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_chunk_document_handles_short_document():
    doc = make_doc(["Just a short sentence."])
    chunks = chunk_document(doc, document_id="doc-4")

    assert len(chunks) == 1
    assert chunks[0].text == "Just a short sentence."
