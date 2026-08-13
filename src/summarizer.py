"""
Document summarization.

Short documents: single LLM call over the full text.
Long documents: simple map-reduce — summarize each chunk, then combine
those partial summaries into one structured final summary. This keeps
the pipeline simple (per spec) rather than a recursive/hierarchical
reduce.
"""
from __future__ import annotations

from src.chunking import Chunk
from src.llm import invoke_with_retry
from src.prompts import MAP_SUMMARY_PROMPT, REDUCE_SUMMARY_PROMPT, SUMMARY_PROMPT

# Rough threshold (characters) above which we switch to map-reduce.
# Kept conservative — well below the model's actual context window — because
# free-tier providers (e.g. Groq at 6000 TPM on llama-3.1-8b-instant) cap
# tokens per MINUTE, not per request: a single "short" document can still
# produce a prompt that alone exceeds the whole budget.
LONG_DOCUMENT_CHAR_THRESHOLD = 3000

# Rough char-count cap (not exact tokenization) used to keep the reduce
# step's combined-partial-summaries prompt from itself becoming too large
# on documents with many chunks.
REDUCE_BATCH_CHAR_LIMIT = 7000


def _reduce_in_batches(llm, summaries: list[str]) -> str:
    """Combine partial summaries into one string, batching through the LLM
    first if the combined text would itself be too large for one call.
    Simple two-level reduce — good enough without over-engineering."""
    combined = "\n\n".join(summaries)
    if len(combined) <= REDUCE_BATCH_CHAR_LIMIT:
        return combined

    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for s in summaries:
        if current_len + len(s) > REDUCE_BATCH_CHAR_LIMIT and current:
            batches.append(current)
            current, current_len = [], 0
        current.append(s)
        current_len += len(s)
    if current:
        batches.append(current)

    batch_summaries = []
    for batch in batches:
        prompt = REDUCE_SUMMARY_PROMPT.format(text="\n\n".join(batch))
        batch_summaries.append(invoke_with_retry(llm, prompt).content)

    return "\n\n".join(batch_summaries)


def summarize_document(llm, full_text: str, chunks: list[Chunk]) -> str:
    if len(full_text) <= LONG_DOCUMENT_CHAR_THRESHOLD:
        prompt = SUMMARY_PROMPT.format(text=full_text)
        return invoke_with_retry(llm, prompt).content

    # Map step: summarize each chunk individually (each call stays small
    # since chunks are already capped by CHUNK_SIZE).
    partial_summaries = []
    for chunk in chunks:
        prompt = MAP_SUMMARY_PROMPT.format(text=chunk.text)
        partial_summaries.append(invoke_with_retry(llm, prompt).content)

    # Reduce step: combine partial summaries into one structured summary,
    # batching first if there are too many to fit in one call.
    combined = _reduce_in_batches(llm, partial_summaries)
    reduce_prompt = REDUCE_SUMMARY_PROMPT.format(text=combined)
    return invoke_with_retry(llm, reduce_prompt).content