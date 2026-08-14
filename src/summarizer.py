"""
Document summarization.

Short documents:
    Full document -> one LLM call -> final summary

Long documents:
    Chunks -> batched summaries -> final reduce -> summary

The batching is specifically designed to reduce the number
of LLM calls and avoid hitting provider rate limits.
"""

from __future__ import annotations

from src.chunking import Chunk
from src.llm import invoke_with_retry
from src.prompts import (
    MAP_SUMMARY_PROMPT,
    REDUCE_SUMMARY_PROMPT,
    SUMMARY_PROMPT,
)


# =============================================================
# CONFIGURATION
# =============================================================

# Documents smaller than this use one LLM request.
LONG_DOCUMENT_CHAR_THRESHOLD = 6000

# Maximum approximate amount of source text sent in one
# map-summary request.
MAP_BATCH_CHAR_LIMIT = 4500

# Maximum approximate size of intermediate summaries
# sent to a reduce request.
REDUCE_BATCH_CHAR_LIMIT = 6000

# Pause between map requests.
#
# This is intentionally conservative for free-tier providers.
# It reduces the chance of sending many requests in a burst.
MAP_REQUEST_DELAY = 2.0


# =============================================================
# CREATE BATCHES
# =============================================================

def _make_batches(
    texts: list[str],
    max_chars: int,
) -> list[list[str]]:
    """
    Group text pieces into batches.

    Example:

        chunk1
        chunk2
        chunk3
             ↓
        batch 1

        chunk4
        chunk5
             ↓
        batch 2
    """

    batches: list[list[str]] = []

    current_batch: list[str] = []
    current_length = 0

    for text in texts:

        text = text.strip()

        if not text:
            continue

        text_length = len(text)

        # If adding this text would make the batch
        # too large, close the current batch.
        if (
            current_batch
            and current_length + text_length
            > max_chars
        ):

            batches.append(
                current_batch
            )

            current_batch = []
            current_length = 0

        current_batch.append(text)
        current_length += text_length

    if current_batch:
        batches.append(
            current_batch
        )

    return batches


# =============================================================
# MAP STEP
# =============================================================

def _map_summaries(
    llm,
    chunks: list[Chunk],
) -> list[str]:
    """
    Create partial summaries.

    Instead of:

        chunk -> LLM
        chunk -> LLM
        chunk -> LLM
        ...

    we do:

        chunk + chunk + chunk -> LLM
        chunk + chunk + chunk -> LLM
        ...
    """

    if not chunks:
        return []

    texts = [
        chunk.text.strip()
        for chunk in chunks
        if chunk.text.strip()
    ]

    if not texts:
        return []

    batches = _make_batches(
        texts,
        MAP_BATCH_CHAR_LIMIT,
    )

    print(
        f"[summary] Created "
        f"{len(batches)} summary batches "
        f"from {len(chunks)} chunks."
    )

    summaries: list[str] = []

    import time

    for index, batch in enumerate(
        batches,
        start=1,
    ):

        combined_text = "\n\n".join(
            batch
        )

        prompt = MAP_SUMMARY_PROMPT.format(
            text=combined_text
        )

        print(
            f"[summary] Processing map batch "
            f"{index}/{len(batches)}..."
        )

        result = invoke_with_retry(
            llm,
            prompt,
        )

        summary = result.content.strip()

        if summary:
            summaries.append(
                summary
            )

        # Small pause between requests.
        #
        # This is NOT a replacement for retry handling.
        # It simply prevents an immediate burst of requests.
        if index < len(batches):
            time.sleep(
                MAP_REQUEST_DELAY
            )

    return summaries


# =============================================================
# REDUCE STEP
# =============================================================

def _reduce_summaries(
    llm,
    summaries: list[str],
) -> str:
    """
    Combine partial summaries into one final summary.

    If all summaries fit into one request:

        partial summaries
              ↓
          final LLM call

    Otherwise:

        partial summaries
              ↓
        smaller reductions
              ↓
        final reduction
    """

    if not summaries:

        return (
            "## Overview\n"
            "No summary could be generated."
        )

    current = summaries

    while True:

        batches = _make_batches(
            current,
            REDUCE_BATCH_CHAR_LIMIT,
        )

        # -----------------------------------------------------
        # Everything fits into one final request.
        # -----------------------------------------------------

        if len(batches) == 1:

            combined = "\n\n".join(
                batches[0]
            )

            prompt = REDUCE_SUMMARY_PROMPT.format(
                text=combined
            )

            print(
                "[summary] Generating final summary..."
            )

            result = invoke_with_retry(
                llm,
                prompt,
            )

            return result.content.strip()

        # -----------------------------------------------------
        # Intermediate reduction required.
        # -----------------------------------------------------

        next_level: list[str] = []

        print(
            f"[summary] Reducing "
            f"{len(batches)} groups..."
        )

        for index, batch in enumerate(
            batches,
            start=1,
        ):

            combined = "\n\n".join(
                batch
            )

            prompt = REDUCE_SUMMARY_PROMPT.format(
                text=combined
            )

            print(
                f"[summary] Reduce batch "
                f"{index}/{len(batches)}..."
            )

            result = invoke_with_retry(
                llm,
                prompt,
            )

            summary = result.content.strip()

            if summary:
                next_level.append(
                    summary
                )

        current = next_level


# =============================================================
# MAIN SUMMARIZATION FUNCTION
# =============================================================

def summarize_document(
    llm,
    full_text: str,
    chunks: list[Chunk],
) -> str:
    """
    Generate a structured document summary.

    Short document:
        full text -> one LLM call

    Long document:
        chunks
          ↓
        batched map summaries
          ↓
        reduce
          ↓
        final structured summary
    """

    if not full_text.strip():

        return (
            "## Overview\n"
            "The document contains no "
            "extractable text."
        )

    # =========================================================
    # SHORT DOCUMENT
    # =========================================================

    if len(full_text) <= LONG_DOCUMENT_CHAR_THRESHOLD:

        print(
            "[summary] Short document. "
            "Using one LLM request."
        )

        prompt = SUMMARY_PROMPT.format(
            text=full_text
        )

        result = invoke_with_retry(
            llm,
            prompt,
        )

        return result.content.strip()

    # =========================================================
    # LONG DOCUMENT
    # =========================================================

    print(
        f"[summary] Long document detected: "
        f"{len(full_text)} characters."
    )

    print(
        f"[summary] Total chunks: "
        f"{len(chunks)}"
    )

    # ---------------------------------------------------------
    # MAP
    # ---------------------------------------------------------

    partial_summaries = _map_summaries(
        llm,
        chunks,
    )

    if not partial_summaries:

        return (
            "## Overview\n"
            "Unable to generate a summary."
        )

    print(
        f"[summary] Generated "
        f"{len(partial_summaries)} partial summaries."
    )

    
    final_summary = _reduce_summaries(
        llm,
        partial_summaries,
    )

    print(
        "[summary] Document summary completed."
    )

    return final_summary
