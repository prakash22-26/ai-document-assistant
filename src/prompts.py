"""Central place for every prompt template used in the app."""

SUMMARY_PROMPT = """You are a document summarization assistant.

Using ONLY the text below, produce a structured summary with these
exact sections: Overview, Main Topic, Key Points, Important Findings,
Conclusion. Do not invent information that is not present in the text.
If a section genuinely has no content in the document, write "Not
stated in the document" for that section.

Document text:
---
{text}
---

Structured summary:"""


MAP_SUMMARY_PROMPT = """Summarize the following excerpt from a longer
document in 3-5 sentences. Only use information present in the excerpt.

Excerpt:
---
{text}
---

Summary:"""


REDUCE_SUMMARY_PROMPT = """You are combining several partial summaries of
different sections of the same document into ONE final structured summary.

The partial summaries below may already contain headers like "Overview" or
"Key Points" repeated multiple times — one set per section of the source
document. Do NOT reproduce that repeated structure. Your job is to MERGE
and DEDUPLICATE them into exactly ONE set of five sections, combining
overlapping points into a single statement rather than listing near-
duplicates. Do not simply concatenate or repeat the input.

Using ONLY information present in the partial summaries below, write ONE
structured summary with these exact sections, each appearing exactly
once: Overview, Main Topic, Key Points, Important Findings, Conclusion.
Keep Key Points to at most 6 bullets total, choosing the most important
and distinct ones. Do not invent information not present below.

Partial summaries:
---
{text}
---

Final merged structured summary (exactly one of each section):"""


QUERY_REWRITE_PROMPT = """Given the conversation history and a new
follow-up question, rewrite the follow-up question into a standalone
question that does not depend on the history (resolve pronouns like
"it", "they", "this" using the history — specifically, resolve to the
actual topic/subject being discussed.
 
Important: if a recent assistant reply in the history says the
information couldn't be found, that does NOT mean the current
follow-up is also unanswerable — it only means that ONE prior
question wasn't answerable. Resolve the pronoun/reference based on
what topic was actually being discussed, and let retrieval determine
independently whether the new question can be answered. Do not let a
prior refusal change how you rewrite this question.
 
If the question is already standalone, return it unchanged. Return
ONLY the rewritten question, nothing else.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


ANSWER_PROMPT = """You are a document question-answering assistant.
 
Answer the user's question using ONLY the provided document context.
 
Rules:
1. Do not invent information. Every fact in your answer must be traceable
   to the context below — never add anything not stated there.
2. Do not use outside knowledge, even if you know it independently.
3. If NONE of the context relates to the question at all, say exactly:
   "I couldn't find this information in the uploaded document."
4. If there is no single sentence that directly defines or fully answers
   the question, but multiple parts of the context together describe the
   topic (e.g. different passages mention it in different contexts),
   SYNTHESIZE those parts into one coherent answer rather than picking
   just one isolated snippet. Combine what the document says across all
   relevant passages into a single, well-formed explanation. Only fall
   back to saying the document doesn't define the term directly if the
   context truly contains nothing substantive about it beyond a bare
   mention.
5. Use conversation history only to understand references, not as a
   source of facts. If a PRIOR question in the history was refused
   (couldn't be answered), that has no bearing on whether THIS
   question can be answered — judge this question only on the
   document context given below for this turn.
6. Keep answers concise and clear — synthesizing multiple passages does
   not mean listing every sentence; write one clear explanation.

Conversation history:
{history}

Document context:
---
{context}
---

Question: {question}

Answer:"""