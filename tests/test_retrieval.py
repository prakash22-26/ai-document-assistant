import shutil
import tempfile

import pytest

from src.chunking import Chunk
from src.vector_store import VectorStore


class FakeEmbeddings:
    """Deterministic fake embedder so tests don't need network/model downloads.

    Encodes text as a tiny vector based on word overlap with a fixed
    vocabulary, which is enough to make similarity search behave
    sensibly for these tests.
    """

    VOCAB = ["dataset", "revenue", "python", "online", "retail", "objective"]

    def _vec(self, text: str) -> list[float]:
        text_lower = text.lower()
        return [1.0 if word in text_lower else 0.0 for word in self.VOCAB]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def temp_store(monkeypatch):
    tmp_dir = tempfile.mkdtemp()
    monkeypatch.setattr("src.vector_store.get_embedding_function", lambda: FakeEmbeddings())
    store = VectorStore(persist_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def make_chunk(chunk_id, text, page, document_id="doc-1"):
    return Chunk(chunk_id=chunk_id, text=text, page=page, source="test.pdf", document_id=document_id)


def test_add_and_search_returns_relevant_chunk(temp_store):
    chunks = [
        make_chunk("chunk_0001", "The dataset used was the Online Retail dataset.", page=4),
        make_chunk("chunk_0002", "The objective of this study was to improve revenue.", page=1),
    ]
    temp_store.add_chunks(chunks)

    results = temp_store.similarity_search("What dataset was used?", document_id="doc-1", top_k=1)

    assert len(results) == 1
    assert results[0]["page"] == 4
    assert "dataset" in results[0]["text"].lower()


def test_search_is_scoped_to_document_id(temp_store):
    temp_store.add_chunks([make_chunk("chunk_0001", "Python revenue dataset info", page=1, document_id="doc-A")])
    temp_store.add_chunks([make_chunk("chunk_0001", "Python revenue dataset info", page=9, document_id="doc-B")])

    results = temp_store.similarity_search("revenue", document_id="doc-A", top_k=5)

    assert all(r for r in results)  # non-empty
    # only doc-A's chunk (page 1) should ever come back for doc-A's search
    pages = temp_store._collection.get(where={"document_id": "doc-A"})
    assert pages["metadatas"][0]["page"] == 1


def test_clear_document_removes_only_that_document(temp_store):
    temp_store.add_chunks([make_chunk("chunk_0001", "keep me", page=1, document_id="keep")])
    temp_store.add_chunks([make_chunk("chunk_0001", "remove me", page=1, document_id="remove")])

    temp_store.clear_document("remove")

    assert temp_store.has_document("keep") is True
    assert temp_store.has_document("remove") is False


def test_similarity_search_empty_store_returns_empty_list(temp_store):
    results = temp_store.similarity_search("anything", document_id="nonexistent")
    assert results == []
