"""
Fast document summarization.

Small/medium documents:
    Full document -> ONE LLM call -> final summary

Large documents:
    Chunks -> batched LLM summaries -> ONE final reduce call

The goal is to minimize the number of LLM requests while keeping
the summary grounded in the uploaded document.
"""

from __future__ import annotations

from src.chunking import Chunk
from src.llm import invoke_with_retry
from src.prompts import (
    MAP_SUMMARY_PROMPT,
    REDUCE_SUMMARY_PROMPT,
    SUMMARY_PROMPT,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Documents below this size use ONE LLM request.
#
# 18,000 characters is roughly a few thousand tokens, leaving
# room for the prompt and model response.
SINGLE_CALL_CHAR_LIMIT = 15_000


# For large documents, several chunks are combined into one
# map request.
#
# Do NOT make this extremely large because the LLM provider
# has token-per-minute and request-size limits.
MAP_BATCH_CHAR_LIMIT = 5_000


# Maximum amount of intermediate summary text sent to the
# final reduce call.
REDUCE_CHAR_LIMIT = 6_000


# ============================================================
# BATCHING
# ============================================================

def _make_batches(
    texts: list[str],
    max_chars: int,
) -> list[list[str]]:
    """
    Group text into batches without exceeding max_chars.
    """

    batches: list[list[str]] = []

    current_batch: list[str] = []
    current_length = 0

    for text in texts:

        text = text.strip()

        if not text:
            continue

        text_length = len(text)

        # If adding this text makes the batch too large,
        # close the current batch.
        if (
            current_batch
            and current_length + text_length > max_chars
        ):
            batches.append(current_batch)

            current_batch = []
            current_length = 0

        current_batch.append(text)
        current_length += text_length

    if current_batch:
        batches.append(current_batch)

    return batches


# ============================================================
# SINGLE-CALL SUMMARY
# ============================================================

def _single_call_summary(
    llm,
    text: str,
) -> str:
    """
    Generate the complete summary using one LLM request.
    """

    print(
        f"[summary] Using single LLM call "
        f"({len(text)} characters)."
    )

    prompt = SUMMARY_PROMPT.format(
        text=text
    )

    result = invoke_with_retry(
        llm,
        prompt,
    )

    return result.content.strip()


# ============================================================
# MAP STEP
# ============================================================

def _map_summaries(
    llm,
    chunks: list[Chunk],
) -> list[str]:
    """
    Summarize groups of chunks.

    Instead of:

        chunk 1 -> LLM
        chunk 2 -> LLM
        chunk 3 -> LLM
        ...

    we do:

        chunk 1 + ... + chunk N -> LLM
        chunk N+1 + ... -> LLM

    This dramatically reduces the number of requests.
    """

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
        f"[summary] Large document."
    )

    print(
        f"[summary] {len(chunks)} chunks "
        f"-> {len(batches)} LLM map calls."
    )

    summaries: list[str] = []

    for index, batch in enumerate(
        batches,
        start=1,
    ):

        combined_text = "\n\n".join(batch)

        print(
            f"[summary] Map "
            f"{index}/{len(batches)} "
            f"({len(combined_text)} characters)"
        )

        prompt = MAP_SUMMARY_PROMPT.format(
            text=combined_text
        )

        result = invoke_with_retry(
            llm,
            prompt,
        )

        summary = result.content.strip()

        if summary:
            summaries.append(summary)

    return summaries


# ============================================================
# REDUCE STEP
# ============================================================

def _reduce_summaries(
    llm,
    summaries: list[str],
) -> str:
    """
    Combine intermediate summaries into one final summary.

    Normally this requires ONE additional LLM call.

    If there are too many intermediate summaries, they are
    reduced in batches first.
    """

    if not summaries:
        return (
            "## Overview\n"
            "Unable to generate a summary."
        )

    # --------------------------------------------------------
    # If all summaries fit in one request, do ONE final call.
    # --------------------------------------------------------

    combined = "\n\n".join(summaries)

    if len(combined) <= REDUCE_CHAR_LIMIT:

        print(
            "[summary] Final reduce: 1 LLM call."
        )

        prompt = REDUCE_SUMMARY_PROMPT.format(
            text=combined
        )

        result = invoke_with_retry(
            llm,
            prompt,
        )

        return result.content.strip()

    # --------------------------------------------------------
    # Too many intermediate summaries.
    # Reduce them in groups.
    # --------------------------------------------------------

    batches = _make_batches(
        summaries,
        REDUCE_CHAR_LIMIT,
    )

    print(
        f"[summary] Intermediate summaries require "
        f"{len(batches)} reduce calls."
    )

    reduced: list[str] = []

    for index, batch in enumerate(
        batches,
        start=1,
    ):

        combined_batch = "\n\n".join(batch)

        print(
            f"[summary] Reduce "
            f"{index}/{len(batches)}"
        )

        prompt = REDUCE_SUMMARY_PROMPT.format(
            text=combined_batch
        )

        result = invoke_with_retry(
            llm,
            prompt,
        )

        summary = result.content.strip()

        if summary:
            reduced.append(summary)

    # --------------------------------------------------------
    # Final reduce.
    # --------------------------------------------------------

    final_text = "\n\n".join(reduced)

    print(
        "[summary] Final reduce call."
    )

    prompt = REDUCE_SUMMARY_PROMPT.format(
        text=final_text
    )

    result = invoke_with_retry(
        llm,
        prompt,
    )

    return result.content.strip()


# ============================================================
# MAIN FUNCTION
# ============================================================

def summarize_document(
    llm,
    full_text: str,
    chunks: list[Chunk],
) -> str:
    """
    Generate a structured document summary.

    Strategy:

    Small/medium:
        full document
            ↓
        ONE LLM call
            ↓
        final summary

    Large:
        chunks
            ↓
        batched map calls
            ↓
        intermediate summaries
            ↓
        ONE/few reduce calls
            ↓
        final summary
    """

    if not full_text.strip():

        return (
            "## Overview\n"
            "The document contains no "
            "extractable text."
        )

    # ========================================================
    # SMALL / MEDIUM DOCUMENT
    # ========================================================

    if len(full_text) <= SINGLE_CALL_CHAR_LIMIT:

        return _single_call_summary(
            llm,
            full_text,
        )

    # ========================================================
    # LARGE DOCUMENT
    # ========================================================

    print(
        f"[summary] Document size: "
        f"{len(full_text)} characters."
    )

    print(
        f"[summary] Total chunks: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # REDUCE
    # --------------------------------------------------------

    final_summary = _reduce_summaries(
        llm,
        partial_summaries,
    )

    print(
        "[summary] Document summary completed."
    )

    return final_summary
