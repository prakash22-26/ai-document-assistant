"""
Retrieval-only evaluation for the RAG system.

IMPORTANT:
This evaluator does NOT call the LLM.

It uses the project's actual:
    document_loader.py
    chunking.py
    vector_store.py
    retriever.py

It evaluates the existing Top-K retrieval using:

    Recall@K
    Precision@K
    Hit@K
    MRR

The ground-truth relevant chunks are taken from:
    evaluation/questions.json

Example questions.json:

{
    "question": "What is the abbreviation for Rectified Linear Unit?",
    "expected_answer": "ReLU",
    "relevant_chunk_ids": [
        "chunk_0013"
    ]
}

The evaluator compares:

    Ground truth relevant chunks
                VS
    Actual Top-K retrieved chunks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# ---------------------------------------------------------
# Allow imports from project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ---------------------------------------------------------
# USE YOUR PROJECT CODE
# ---------------------------------------------------------

from src.document_loader import load_pdf
from src.chunking import chunk_document
from src.vector_store import VectorStore
from src.retriever import retrieve


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

QUESTIONS_PATH = (
    Path(__file__).parent / "questions.json"
)

RESULTS_PATH = (
    Path(__file__).parent / "results.json"
)


# ---------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------

def calculate_recall_at_k(
    relevant_ids: set[str],
    retrieved_ids: list[str]
) -> float | None:

    if not relevant_ids:
        return None

    retrieved_set = set(
        retrieved_ids
    )

    found = (
        relevant_ids
        & retrieved_set
    )

    return (
        len(found)
        / len(relevant_ids)
    )


def calculate_precision_at_k(
    relevant_ids: set[str],
    retrieved_ids: list[str]
) -> float:

    if not retrieved_ids:
        return 0.0

    retrieved_set = set(
        retrieved_ids
    )

    found = (
        relevant_ids
        & retrieved_set
    )

    return (
        len(found)
        / len(retrieved_ids)
    )


def calculate_hit_at_k(
    relevant_ids: set[str],
    retrieved_ids: list[str]
) -> float | None:

    if not relevant_ids:
        return None

    retrieved_set = set(
        retrieved_ids
    )

    return (
        1.0
        if relevant_ids & retrieved_set
        else 0.0
    )


def calculate_mrr(
    relevant_ids: set[str],
    retrieved_ids: list[str]
) -> float | None:

    if not relevant_ids:
        return None

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if chunk_id in relevant_ids:

            return 1.0 / rank

    return 0.0


# ---------------------------------------------------------
# Get chunk IDs
# ---------------------------------------------------------

def get_chunk_ids(
    documents: list[dict]
) -> list[str]:

    ids = []

    for document in documents:

        chunk_id = document.get(
            "chunk_id"
        )

        if chunk_id:
            ids.append(
                chunk_id
            )

    return ids


# ---------------------------------------------------------
# Evaluate one question
# ---------------------------------------------------------

def evaluate_question(
    vector_store: VectorStore,
    question_item: dict,
    document_id: str
) -> dict:

    question = question_item[
        "question"
    ]

    relevant_ids = set(
        question_item.get(
            "relevant_chunk_ids",
            []
        )
    )

    print(
        "\n" + "-" * 70
    )

    print(
        f"Question:\n{question}"
    )

    print(
        "\nGround-truth relevant chunks:"
    )

    if relevant_ids:

        for chunk_id in sorted(
            relevant_ids
        ):

            print(
                f"  - {chunk_id}"
            )

    else:

        print(
            "  NONE"
        )

    # -----------------------------------------------------
    # IMPORTANT
    #
    # This calls YOUR project's existing retrieval.
    #
    # No LLM.
    # No summarization.
    # No answer generation.
    # -----------------------------------------------------

    retrieved_documents = retrieve(
        vector_store,
        query=question,
        document_id=document_id
    )

    retrieved_ids = get_chunk_ids(
        retrieved_documents
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    recall = calculate_recall_at_k(
        relevant_ids,
        retrieved_ids
    )

    precision = calculate_precision_at_k(
        relevant_ids,
        retrieved_ids
    )

    hit = calculate_hit_at_k(
        relevant_ids,
        retrieved_ids
    )

    mrr = calculate_mrr(
        relevant_ids,
        retrieved_ids
    )

    # -----------------------------------------------------
    # Display retrieved chunks
    # -----------------------------------------------------

    print(
        "\nActual Top-K retrieved:"
    )

    if retrieved_ids:

        for rank, chunk_id in enumerate(
            retrieved_ids,
            start=1
        ):

            if chunk_id in relevant_ids:

                marker = "YES"

            else:

                marker = "NO"

            print(
                f"  {rank}. "
                f"{chunk_id} "
                f"[relevant={marker}]"
            )

    else:

        print(
            "  No chunks retrieved."
        )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    print(
        "\nMetrics:"
    )

    if recall is not None:

        print(
            f"  Recall@K    : "
            f"{recall * 100:.2f}%"
        )

    else:

        print(
            "  Recall@K    : N/A"
        )

    print(
        f"  Precision@K : "
        f"{precision * 100:.2f}%"
    )

    if hit is not None:

        print(
            f"  Hit@K       : "
            f"{hit * 100:.2f}%"
        )

    else:

        print(
            "  Hit@K       : N/A"
        )

    if mrr is not None:

        print(
            f"  MRR         : "
            f"{mrr:.4f}"
        )

    else:

        print(
            "  MRR         : N/A"
        )

    # -----------------------------------------------------
    # Which retrieved chunks were actually relevant?
    # -----------------------------------------------------

    relevant_retrieved = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in relevant_ids
    ]

    return {
        "question": question,

        "expected_answer":
            question_item.get(
                "expected_answer"
            ),

        "relevant_chunk_ids":
            sorted(
                relevant_ids
            ),

        "retrieved_chunk_ids":
            retrieved_ids,

        "relevant_retrieved_chunk_ids":
            relevant_retrieved,

        "recall_at_k":
            recall,

        "precision_at_k":
            precision,

        "hit_at_k":
            hit,

        "mrr":
            mrr
    }


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def run_evaluation(
    pdf_path: str
) -> None:

    # -----------------------------------------------------
    # Load questions
    # -----------------------------------------------------

    print(
        "\nLoading evaluation questions..."
    )

    questions = json.loads(
        QUESTIONS_PATH.read_text(
            encoding="utf-8"
        )
    )

    print(
        f"Evaluation questions: "
        f"{len(questions)}"
    )

    # -----------------------------------------------------
    # Load PDF
    # -----------------------------------------------------

    print(
        "\nLoading PDF..."
    )

    with open(
        pdf_path,
        "rb"
    ) as file:

        file_bytes = file.read()

    document = load_pdf(
        file_bytes,
        Path(pdf_path).name
    )

    print(
        f"Document: "
        f"{Path(pdf_path).name}"
    )

    # -----------------------------------------------------
    # Create chunks using YOUR project
    # -----------------------------------------------------

    print(
        "\nCreating chunks using "
        "src/chunking.py..."
    )

    document_id = (
        document.document_hash
    )

    chunks = chunk_document(
        document,
        document_id
    )

    print(
        f"Total project chunks: "
        f"{len(chunks)}"
    )

    # -----------------------------------------------------
    # Create YOUR vector store
    # -----------------------------------------------------

    print(
        "\nCreating vector store..."
    )

    vector_store = VectorStore()

    # -----------------------------------------------------
    # Add chunks using YOUR project code
    # -----------------------------------------------------

    print(
        "\nAdding project chunks "
        "to vector store..."
    )

    vector_store.clear_document(
        document_id
    )

    vector_store.add_chunks(
        chunks
    )

    print(
        "Chunks added successfully."
    )

    # -----------------------------------------------------
    # Evaluation counters
    # -----------------------------------------------------

    results = []

    retrieval_questions = 0

    total_recall = 0.0
    total_precision = 0.0
    total_hit = 0.0
    total_mrr = 0.0

    # -----------------------------------------------------
    # Evaluate every question
    # -----------------------------------------------------

    for index, question_item in enumerate(
        questions,
        start=1
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"QUESTION "
            f"{index}/{len(questions)}"
        )

        print(
            "=" * 70
        )

        result = evaluate_question(
            vector_store,
            question_item,
            document_id
        )

        results.append(
            result
        )

        # -------------------------------------------------
        # Only questions with ground-truth relevant
        # chunks contribute to retrieval metrics.
        # -------------------------------------------------

        relevant_ids = set(
            result[
                "relevant_chunk_ids"
            ]
        )

        if relevant_ids:

            retrieval_questions += 1

            total_recall += (
                result["recall_at_k"]
                or 0.0
            )

            total_precision += (
                result["precision_at_k"]
                or 0.0
            )

            total_hit += (
                result["hit_at_k"]
                or 0.0
            )

            total_mrr += (
                result["mrr"]
                or 0.0
            )

    # -----------------------------------------------------
    # Overall metrics
    # -----------------------------------------------------

    if retrieval_questions:

        average_recall = (
            total_recall
            / retrieval_questions
        )

        average_precision = (
            total_precision
            / retrieval_questions
        )

        average_hit = (
            total_hit
            / retrieval_questions
        )

        average_mrr = (
            total_mrr
            / retrieval_questions
        )

    else:

        average_recall = 0.0
        average_precision = 0.0
        average_hit = 0.0
        average_mrr = 0.0

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = {

        "document":
            Path(pdf_path).name,

        "total_questions":
            len(questions),

        "retrieval_questions":
            retrieval_questions,

        "retrieval": {

            "recall_at_k_pct":
                round(
                    average_recall * 100,
                    2
                ),

            "precision_at_k_pct":
                round(
                    average_precision * 100,
                    2
                ),

            "hit_at_k_pct":
                round(
                    average_hit * 100,
                    2
                ),

            "mrr":
                round(
                    average_mrr,
                    4
                )
        },

        "results":
            results
    }

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    RESULTS_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print(
        "\n\n" + "=" * 70
    )

    print(
        "RETRIEVAL EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nQuestions evaluated: "
        f"{len(questions)}"
    )

    print(
        f"Questions with ground truth: "
        f"{retrieval_questions}"
    )

    print(
        "\nOverall Retrieval Metrics"
    )

    print(
        "--------------------------"
    )

    print(
        f"Recall@K: "
        f"{average_recall * 100:.2f}%"
    )

    print(
        f"Precision@K: "
        f"{average_precision * 100:.2f}%"
    )

    print(
        f"Hit@K: "
        f"{average_hit * 100:.2f}%"
    )

    print(
        f"MRR: "
        f"{average_mrr:.4f}"
    )

    print(
        f"\nResults saved to:"
    )

    print(
        RESULTS_PATH
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python evaluation/evaluate.py "
            "<path-to-pdf>"
        )

        sys.exit(1)

    run_evaluation(
        sys.argv[1]
    )