"""
Fast document summarization.

Short documents:
    Full document -> 1 LLM call

Long documents:
    Representative document sections -> 2 LLM calls
    -> 1 final reduce call

The number of LLM calls is deliberately bounded so that
large PDFs do not become extremely slow or hit Groq
rate limits.
"""

from __future__ import annotations

from src.chunking import Chunk
from src.llm import invoke_with_retry
from src.prompts import (
    REDUCE_SUMMARY_PROMPT,
    SUMMARY_PROMPT,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Small documents are summarized directly.
SHORT_DOCUMENT_LIMIT = 8000

# Maximum characters sent to each map call.
MAP_INPUT_LIMIT = 6500

# Maximum number of map calls.
# Therefore maximum total summary calls = 3.
MAX_MAP_CALLS = 2


# ---------------------------------------------------------
# Build representative document sections
# ---------------------------------------------------------

def _select_representative_chunks(
    chunks: list[Chunk],
) -> list[str]:
    """
    Select representative chunks from the document.

    Instead of sending every chunk to the LLM, select
    chunks distributed throughout the document.

    This keeps summarization fast for large PDFs.
    """

    valid_chunks = [
        chunk.text.strip()
        for chunk in chunks
        if chunk.text.strip()
    ]

    if not valid_chunks:
        return []

    # If the document is already small, keep everything.
    total_chars = sum(
        len(text)
        for text in valid_chunks
    )

    if total_chars <= MAP_INPUT_LIMIT * MAX_MAP_CALLS:
        return valid_chunks

    # Select chunks distributed across the document.
    selected = []

    step = len(valid_chunks) / (
        MAX_MAP_CALLS * 5
    )

    index = 0.0

    while (
        int(index) < len(valid_chunks)
        and len(selected) < MAX_MAP_CALLS * 5
    ):
        selected.append(
            valid_chunks[int(index)]
        )
        index += step

    return selected


# ---------------------------------------------------------
# Create map batches
# ---------------------------------------------------------

def _create_map_batches(
    texts: list[str],
) -> list[str]:
    """
    Create at most MAX_MAP_CALLS batches.
    """

    if not texts:
        return []

    # First create a single combined document.
    combined = "\n\n".join(texts)

    # If it fits into one request, use one call.
    if len(combined) <= MAP_INPUT_LIMIT:
        return [combined]

    # Split approximately evenly between the
    # maximum number of LLM calls.
    target_size = min(
        MAP_INPUT_LIMIT,
        len(combined) // MAX_MAP_CALLS + 1,
    )

    batches = []

    current = []

    current_length = 0

    for text in texts:

        if (
            current
            and current_length + len(text)
            > target_size
        ):
            batches.append(
                "\n\n".join(current)
            )

            current = []
            current_length = 0

        current.append(text)
        current_length += len(text)

    if current:
        batches.append(
            "\n\n".join(current)
        )

    # Never exceed MAX_MAP_CALLS.
    if len(batches) > MAX_MAP_CALLS:

        merged = []

        split_size = (
            len(batches)
            // MAX_MAP_CALLS
        )

        for i in range(MAX_MAP_CALLS):

            start = i * split_size

            if i == MAX_MAP_CALLS - 1:
                end = len(batches)
            else:
                end = (
                    (i + 1)
                    * split_size
                )

            merged.append(
                "\n\n".join(
                    batches[start:end]
                )
            )

        batches = merged

    return batches


# ---------------------------------------------------------
# Main summarization
# ---------------------------------------------------------

def summarize_document(
    llm,
    full_text: str,
    chunks: list[Chunk],
) -> str:

    if not full_text.strip():

        return (
            "## Overview\n"
            "The document contains no "
            "extractable text."
        )

    # =====================================================
    # SHORT DOCUMENT
    # =====================================================

    if len(full_text) <= SHORT_DOCUMENT_LIMIT:

        print(
            "[summary] Short document - "
            "using 1 LLM call."
        )

        prompt = SUMMARY_PROMPT.format(
            text=full_text
        )

        result = invoke_with_retry(
            llm,
            prompt,
        )

        return result.content.strip()

    # =====================================================
    # LONG DOCUMENT
    # =====================================================

    print(
        f"[summary] Long document: "
        f"{len(full_text)} characters"
    )

    print(
        f"[summary] Total chunks: "
        f"{len(chunks)}"
    )

    # -----------------------------------------------------
    # Select representative content
    # -----------------------------------------------------

    selected_chunks = (
        _select_representative_chunks(
            chunks
        )
    )

    print(
        f"[summary] Selected "
        f"{len(selected_chunks)} "
        f"representative chunks."
    )

    # -----------------------------------------------------
    # Create maximum 2 map batches
    # -----------------------------------------------------

    batches = _create_map_batches(
        selected_chunks
    )

    print(
        f"[summary] Using "
        f"{len(batches)} map calls."
    )

    # -----------------------------------------------------
    # MAP
    # -----------------------------------------------------

    partial_summaries = []

    for index, batch in enumerate(
        batches,
        start=1,
    ):

        prompt = f"""
Summarize the following section of a document.

Focus on:
- Main topic
- Important facts
- Key findings
- Important numbers and dates
- Important conclusions

Use only the supplied document content.
Do not invent information.

DOCUMENT SECTION:

{batch}
"""

        print(
            f"[summary] Map call "
            f"{index}/{len(batches)}"
        )

        result = invoke_with_retry(
            llm,
            prompt,
        )

        summary = result.content.strip()

        if summary:
            partial_summaries.append(
                summary
            )

    if not partial_summaries:

        return (
            "## Overview\n"
            "Unable to generate a summary."
        )

    # =====================================================
    # FINAL REDUCE
    # =====================================================

    combined = "\n\n".join(
        partial_summaries
    )

    print(
        "[summary] Final reduce call."
    )

    prompt = REDUCE_SUMMARY_PROMPT.format(
        text=combined
    )

    result = invoke_with_retry(
        llm,
        prompt,
    )

    print(
        "[summary] Summary completed."
    )

    return result.content.strip()
