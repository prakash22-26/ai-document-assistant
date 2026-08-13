import io

import pytest
from pypdf import PdfWriter

from src.document_loader import DocumentLoadError, compute_document_hash, load_pdf


def _escape_pdf_text(s: str) -> str:
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def make_pdf_bytes(pages_text: list[str]) -> bytes:
    """Hand-build a minimal valid multi-page PDF with real extractable text.

    Deliberately avoids any extra dependency (e.g. reportlab) so this test
    suite has no requirements beyond requirements.txt.
    """
    n_pages = len(pages_text)
    catalog_num = 1
    pages_num = 2
    font_num = 3
    page_nums = [4 + 2 * i for i in range(n_pages)]
    content_nums = [5 + 2 * i for i in range(n_pages)]

    objs: dict[int, bytes] = {}
    objs[catalog_num] = f"{catalog_num} 0 obj\n<< /Type /Catalog /Pages {pages_num} 0 R >>\nendobj\n".encode()
    kids = " ".join(f"{p} 0 R" for p in page_nums)
    objs[pages_num] = (
        f"{pages_num} 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>\nendobj\n".encode()
    )
    objs[font_num] = (
        f"{font_num} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode()
    )

    for i, text in enumerate(pages_text):
        page_num = page_nums[i]
        content_num = content_nums[i]
        stream = f"BT /F1 12 Tf 72 720 Td ({_escape_pdf_text(text)}) Tj ET".encode()
        objs[content_num] = (
            f"{content_num} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream\nendobj\n"
        )
        objs[page_num] = (
            f"{page_num} 0 obj\n<< /Type /Page /Parent {pages_num} 0 R "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> "
            f"/MediaBox [0 0 612 792] /Contents {content_num} 0 R >>\nendobj\n"
        ).encode()

    header = b"%PDF-1.4\n"
    body = bytearray()
    offsets: dict[int, int] = {}
    for num in sorted(objs.keys()):
        offsets[num] = len(header) + len(body)
        body += objs[num]

    xref_offset = len(header) + len(body)
    max_num = max(objs.keys())
    xref = f"xref\n0 {max_num + 1}\n0000000000 65535 f \n".encode()
    for num in range(1, max_num + 1):
        xref += f"{offsets.get(num, 0):010d} 00000 n \n".encode()

    trailer = f"trailer\n<< /Size {max_num + 1} /Root {catalog_num} 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()

    return header + bytes(body) + xref + trailer


def test_load_pdf_extracts_text_and_pages():
    pdf_bytes = make_pdf_bytes(["Hello world page one", "Second page content here"])
    doc = load_pdf(pdf_bytes, "sample.pdf")

    assert doc.filename == "sample.pdf"
    assert len(doc.pages) == 2
    assert doc.pages[0].page == 1
    assert doc.pages[1].page == 2
    assert not doc.is_empty


def test_load_pdf_rejects_non_pdf_extension():
    with pytest.raises(DocumentLoadError):
        load_pdf(b"not a real pdf", "notes.txt")


def test_load_pdf_rejects_empty_bytes():
    with pytest.raises(DocumentLoadError):
        load_pdf(b"", "empty.pdf")


def test_load_pdf_rejects_corrupt_pdf():
    with pytest.raises(DocumentLoadError):
        load_pdf(b"%PDF-1.4 this is not actually a valid pdf structure", "bad.pdf")


def test_compute_document_hash_is_stable():
    data = b"same bytes"
    assert compute_document_hash(data) == compute_document_hash(data)
    assert compute_document_hash(data) != compute_document_hash(b"different bytes")


def test_load_pdf_blank_pages_raise_scanned_pdf_error():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    blank_pdf = buf.getvalue()

    with pytest.raises(DocumentLoadError):
        load_pdf(blank_pdf, "scanned.pdf")