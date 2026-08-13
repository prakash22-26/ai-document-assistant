from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from src.document_loader import load_pdf
from src.chunking import chunk_document
from src.llm import get_llm, invoke_with_retry


OUTPUT_PATH = (
    Path(__file__).parent / "questions.json"
)

# Number of evaluation questions
MAX_QUESTIONS = 30


PROMPT = """
You are creating a retrieval evaluation dataset for a RAG system.

Read the document chunk below.

Generate ONE factual question that can be answered directly
and completely from this chunk.

Also provide the answer using ONLY information contained
in this chunk.

IMPORTANT:

- The question must depend on information in this chunk.
- Do not use outside knowledge.
- Do not create a question whose answer requires another chunk.
- Keep the question clear and specific.
- Do not mention the chunk in the question.
- The expected answer must be directly supported by the chunk.
- Prefer questions whose answer is explicitly stated in the chunk.
- Do not create vague questions.

Return ONLY valid JSON in exactly this format:

{{
    "question": "...",
    "expected_answer": "..."
}}

DOCUMENT CHUNK:

{chunk_text}
"""


def generate_question(
    llm,
    chunk_text: str
):
    """Generate one question from one project chunk."""

    prompt = PROMPT.format(
        chunk_text=chunk_text
    )

    response = invoke_with_retry(
        llm,
        prompt
    )

    content = response.content.strip()

    # Remove markdown code fences if the LLM adds them
    if content.startswith("```"):
        content = content.replace(
            "```json",
            ""
        )
        content = content.replace(
            "```",
            ""
        )
        content = content.strip()

    try:
        data = json.loads(content)

    except json.JSONDecodeError:
        print(
            "WARNING: LLM returned invalid JSON."
        )
        return None

    if not isinstance(data, dict):
        return None

    question = data.get("question")
    answer = data.get("expected_answer")

    if not question or not answer:
        return None

    return {
        "question": question.strip(),
        "expected_answer": answer.strip()
    }


def select_chunks(
    chunks,
    max_questions: int
):
    """
    Select chunks distributed throughout the document.

    This avoids generating all questions from only
    the first 30 chunks.
    """

    if len(chunks) <= max_questions:
        return chunks

    selected = []

    step = len(chunks) / max_questions

    for i in range(max_questions):
        index = int(i * step)

        selected.append(
            chunks[index]
        )

    return selected


def main(pdf_path: str):

    print("\nLoading document...")

    with open(
        pdf_path,
        "rb"
    ) as file:

        file_bytes = file.read()

    # IMPORTANT:
    # Uses your project's document loader.
    document = load_pdf(
        file_bytes,
        Path(pdf_path).name
    )

    print(
        f"Document: {Path(pdf_path).name}"
    )

    print("\nCreating chunks...")

    # IMPORTANT:
    # Uses your project's actual chunking implementation.
    document_id = document.document_hash

    chunks = chunk_document(
        document,
        document_id
    )

    print(
        f"Total project chunks: {len(chunks)}"
    )

    llm = get_llm()

    selected_chunks = select_chunks(
        chunks,
        MAX_QUESTIONS
    )

    print(
        f"\nGenerating questions from "
        f"{len(selected_chunks)} chunks..."
    )

    questions = []

    for index, chunk in enumerate(
        selected_chunks,
        start=1
    ):

        print(
            f"\n[{index}/{len(selected_chunks)}] "
            f"Processing {chunk.chunk_id}"
        )

        result = generate_question(
            llm,
            chunk.text
        )

        if result is None:

            print(
                "Could not generate valid question."
            )

            continue

        # IMPORTANT:
        #
        # The chunk used to generate the question
        # is the ground-truth relevant chunk for
        # this single-chunk evaluation question.
        #
        # We do NOT run another LLM over all 128
        # chunks to create noisy labels.
        evaluation_item = {

            "question":
                result["question"],

            "expected_answer":
                result["expected_answer"],

            "relevant_chunk_ids": [
                chunk.chunk_id
            ],

            "source_page":
                chunk.page,

            "source_chunk_id":
                chunk.chunk_id
        }

        questions.append(
            evaluation_item
        )

        print(
            f"Question: {result['question']}"
        )

        print(
            f"Ground-truth chunk: "
            f"{chunk.chunk_id}"
        )

    # -------------------------------------------------
    # Add one out-of-document question
    # -------------------------------------------------

    questions.append(
        {
            "question":
                "Who is the president "
                "of the United States?",

            "expected_answer":
                "NOT_IN_DOCUMENT",

            "relevant_chunk_ids": [],

            "source_page": None,

            "source_chunk_id": None
        }
    )

    # -------------------------------------------------
    # Save questions
    # -------------------------------------------------

    OUTPUT_PATH.write_text(
        json.dumps(
            questions,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        "\n" + "=" * 60
    )

    print("DONE")

    print(
        "=" * 60
    )

    print(
        f"\nGenerated questions: "
        f"{len(questions)}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python evaluation/generate_questions.py "
            "<path-to-pdf>"
        )

        sys.exit(1)

    main(sys.argv[1])