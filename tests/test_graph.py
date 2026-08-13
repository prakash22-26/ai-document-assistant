from unittest.mock import MagicMock

from src.graph import NO_CONTEXT_ANSWER, build_graph


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


def make_fake_llm(rewrite_response: str, answer_response: str):
    """First .invoke() call in a turn is the rewrite, second is the answer."""
    llm = MagicMock()
    llm.invoke.side_effect = [
        FakeLLMResponse(rewrite_response),
        FakeLLMResponse(answer_response),
    ]
    return llm


def test_graph_answers_with_retrieved_context():
    llm = make_fake_llm(
        rewrite_response="What dataset was used?",
        answer_response="They used the Online Retail dataset.",
    )

    vector_store = MagicMock()

    vector_store.similarity_search.return_value = [
        {
            "text": "The Online Retail dataset was used.",
            "page": 4,
            "source": "paper.pdf",
            "chunk_id": "chunk_0001",
            "distance": 0.1,
        }
    ]

    graph = build_graph(llm, vector_store)

    result = graph.invoke(
        {
            "question": "What dataset was used?",
            "standalone_question": "",
            "chat_history": [
                {
                    "role": "user",
                    "content": "hi",
                }
            ],
            "document_id": "doc-1",
            "retrieved_documents": [],
            "answer": "",
            "sources": [],
        }
    )

    assert result["answer"] == (
        "They used the Online Retail dataset."
    )

    assert result["sources"] == [4]


def test_graph_no_context_behavior_when_no_chunks_found():
    llm = MagicMock()

    llm.invoke.return_value = FakeLLMResponse(
        "irrelevant, retrieve short-circuits before 2nd call"
    )

    vector_store = MagicMock()

    vector_store.similarity_search.return_value = []

    graph = build_graph(
        llm,
        vector_store,
    )

    result = graph.invoke(
        {
            "question": (
                "What is the airspeed velocity "
                "of an unladen swallow?"
            ),
            "standalone_question": "",
            "chat_history": [],
            "document_id": "doc-1",
            "retrieved_documents": [],
            "answer": "",
            "sources": [],
        }
    )

    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["sources"] == []


def test_graph_skips_rewrite_on_first_turn():
    llm = MagicMock()

    llm.invoke.return_value = FakeLLMResponse(
        "The main objective is efficiency."
    )

    vector_store = MagicMock()

    vector_store.similarity_search.return_value = [
        {
            "text": "efficiency context",
            "page": 1,
            "source": "p.pdf",
            "chunk_id": "c1",
            "distance": 0.05,
        }
    ]

    graph = build_graph(
        llm,
        vector_store,
    )

    result = graph.invoke(
        {
            "question": "What is the main objective?",
            "standalone_question": "",
            "chat_history": [],
            "document_id": "doc-1",
            "retrieved_documents": [],
            "answer": "",
            "sources": [],
        }
    )

    # Only generate_answer calls the LLM.
    # No rewrite happens because there is no history.
    assert llm.invoke.call_count == 1

    assert (
        result["standalone_question"]
        == "What is the main objective?"
    )


def test_graph_no_sources_when_llm_refuses_despite_retrieved_chunks():
    """
    Retrieval can return a chunk even when it is irrelevant
    to the question.

    If the LLM correctly refuses to answer, the refusal must
    not carry a misleading page citation.
    """

    llm = make_fake_llm(
        rewrite_response="Who is the president of France?",
        answer_response=NO_CONTEXT_ANSWER,
    )

    vector_store = MagicMock()

    vector_store.similarity_search.return_value = [
        {
            "text": "unrelated content",
            "page": 1,
            "source": "p.pdf",
            "chunk_id": "c1",
            "distance": 0.9,
        }
    ]

    graph = build_graph(
        llm,
        vector_store,
    )

    result = graph.invoke(
        {
            "question": "Who is the president of France?",
            "standalone_question": "",
            "chat_history": [
                {
                    "role": "user",
                    "content": "hi",
                }
            ],
            "document_id": "doc-1",
            "retrieved_documents": [],
            "answer": "",
            "sources": [],
        }
    )

    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["sources"] == []


def test_graph_expands_short_acronym_query_before_retrieval():
    """
    A bare 1-2 word query should be expanded before hitting
    the vector store because short queries can embed poorly.
    """

    llm = MagicMock()

    llm.invoke.return_value = FakeLLMResponse(
        "Graph Neural Network is a type of neural network."
    )

    vector_store = MagicMock()

    vector_store.similarity_search.return_value = [
        {
            "text": "GNN context",
            "page": 5,
            "source": "p.pdf",
            "chunk_id": "c1",
            "distance": 0.1,
        }
    ]

    graph = build_graph(
        llm,
        vector_store,
    )

    graph.invoke(
        {
            "question": "gnn",
            "standalone_question": "",
            "chat_history": [],
            "document_id": "doc-1",
            "retrieved_documents": [],
            "answer": "",
            "sources": [],
        }
    )

    called_query = (
        vector_store
        .similarity_search
        .call_args
        .kwargs["query"]
    )

    assert called_query != "gnn"
    assert "gnn" in called_query.lower()
    assert len(called_query.split()) > 2


def test_graph_does_not_expand_normal_length_query():
    """
    A full question should be sent to retrieval unchanged.
    """

    llm = MagicMock()

    llm.invoke.return_value = FakeLLMResponse(
        "They used the Online Retail dataset."
    )

    vector_store = MagicMock()

    vector_store.similarity_search.return_value = [
        {
            "text": "dataset context",
            "page": 4,
            "source": "p.pdf",
            "chunk_id": "c1",
            "distance": 0.1,
        }
    ]

    graph = build_graph(
        llm,
        vector_store,
    )

    graph.invoke(
        {
            "question": "What dataset was used?",
            "standalone_question": "",
            "chat_history": [],
            "document_id": "doc-1",
            "retrieved_documents": [],
            "answer": "",
            "sources": [],
        }
    )

    called_query = (
        vector_store
        .similarity_search
        .call_args
        .kwargs["query"]
    )

    assert called_query == "What dataset was used?"


def test_graph_does_not_rewrite_bare_word_with_history():
    """
    A standalone one-word query should NOT be rewritten using
    previous conversation history.

    This prevents an unrelated query such as 'humanbody' from
    being incorrectly associated with the previous conversation.
    """

    llm = MagicMock()

    # Only answer generation should call the LLM.
    llm.invoke.return_value = FakeLLMResponse(
        "Fingerprints are important for identification."
    )

    vector_store = MagicMock()

    vector_store.similarity_search.return_value = [
        {
            "text": (
                "Fingerprints are important for identification."
            ),
            "page": 2,
            "source": "p.pdf",
            "chunk_id": "c1",
            "distance": 0.1,
        }
    ]

    graph = build_graph(
        llm,
        vector_store,
    )

    result = graph.invoke(
        {
            "question": "fingerprint",
            "standalone_question": "",
            "chat_history": [
                {
                    "role": "user",
                    "content": "What are fingerprints?",
                },
                {
                    "role": "assistant",
                    "content": (
                        "Fingerprints are unique patterns "
                        "used for identification."
                    ),
                },
            ],
            "document_id": "doc-1",
            "retrieved_documents": [],
            "answer": "",
            "sources": [],
            "answerable": False,
        }
    )

    # One LLM call only:
    # generate_answer.
    assert llm.invoke.call_count == 1

    # The bare word must remain unchanged.
    assert result["standalone_question"] == "fingerprint"

    assert result["answer"] == (
        "Fingerprints are important for identification."
    )

    assert result["sources"] == [2]

    # The short-query expansion should still happen
    # before retrieval.
    called_query = (
        vector_store
        .similarity_search
        .call_args
        .kwargs["query"]
    )

    assert called_query == (
        "What is fingerprint? "
        "Explain fingerprint in detail."
    )


def test_followup_works_after_multiple_not_found_questions():
    """
    A later follow-up should still use the previous successful
    conversation even after unrelated questions.

    The application stores only successful conversations in
    chat_history.
    """

    llm = MagicMock()

    llm.invoke.side_effect = [
        # First call: query rewrite
        FakeLLMResponse(
            "Why are fingerprints important?"
        ),

        # Second call: final answer
        FakeLLMResponse(
            "Fingerprints are important because "
            "they are useful for identification."
        ),
    ]

    vector_store = MagicMock()

    vector_store.similarity_search.return_value = [
        {
            "text": (
                "Fingerprints are important because "
                "they are useful for identification."
            ),
            "page": 12,
            "source": "paper.pdf",
            "chunk_id": "chunk_0012",
            "distance": 0.1,
        }
    ]

    graph = build_graph(
        llm,
        vector_store,
    )

    result = graph.invoke(
        {
            "question": "what is important?",
            "standalone_question": "",

            # Only successful conversation is provided.
            # NOT FOUND turns are excluded by app.py.
            "chat_history": [
                {
                    "role": "user",
                    "content": "What are fingerprints?",
                },
                {
                    "role": "assistant",
                    "content": (
                        "Fingerprints are unique patterns "
                        "used for identification."
                    ),
                },
            ],

            "document_id": "doc-1",
            "retrieved_documents": [],
            "answer": "",
            "sources": [],
            "answerable": False,
        }
    )

    assert result["answer"] == (
        "Fingerprints are important because "
        "they are useful for identification."
    )

    assert result["sources"] == [12]

    assert result["answerable"] is True

    called_query = (
        vector_store
        .similarity_search
        .call_args
        .kwargs["query"]
    )

    assert called_query == (
        "Why are fingerprints important?"
    )