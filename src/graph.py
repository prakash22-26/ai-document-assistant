"""
LangGraph conversational RAG workflow.

    START -> understand_question -> retrieve -> generate_answer -> END

Three nodes, one linear path — deliberately not a multi-agent system.
The LLM and vector_store are injected via a factory (build_graph) rather
than imported globally, which makes the graph trivially testable with a
fake/mock LLM.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.prompts import ANSWER_PROMPT, QUERY_REWRITE_PROMPT
from src.retriever import retrieve
from src.state import GraphState
from src.llm import invoke_with_retry
from src.vector_store import VectorStore

NO_CONTEXT_ANSWER = "I couldn't find this information in the uploaded document."


def _format_history(chat_history: list[dict]) -> str:
    if not chat_history:
        return "(no previous conversation)"
    lines = [f"{turn['role']}: {turn['content']}" for turn in chat_history]
    return "\n".join(lines)


def build_graph(llm, vector_store: VectorStore):
    """Returns a compiled LangGraph app. `llm` must expose `.invoke(str) -> obj with .content`."""

    def understand_question(state: GraphState) -> GraphState:
        history = state.get("chat_history", [])
        raw_question = state["question"]
        if not history or len(raw_question.split()) <= 2:
            standalone = raw_question
        else:
            prompt = QUERY_REWRITE_PROMPT.format(
                history=_format_history(history), question=raw_question
            )
            standalone = invoke_with_retry(llm, prompt).content.strip()
        return {**state, "standalone_question": standalone}

    def retrieve_node(state: GraphState) -> GraphState:
        query = state["standalone_question"]
        if len(query.split()) <= 2:
            query = f"What is {query}? Explain {query} in detail."

        hits = retrieve(
            vector_store,
            query=query,
            document_id=state["document_id"],
        )

        return {**state, "retrieved_documents": hits}

    def generate_answer(state: GraphState) -> GraphState:
        hits = state.get("retrieved_documents", [])
        if not hits:
            return {**state, "answer": NO_CONTEXT_ANSWER, "sources": [],"answerable": False,}

        context = "\n\n".join(f"[Page {h['page']}] {h['text']}" for h in hits)
        prompt = ANSWER_PROMPT.format(
            history=_format_history(state.get("chat_history", [])),
            context=context,
            question=state["standalone_question"],
        )
        answer = invoke_with_retry(llm, prompt).content.strip()
        if NO_CONTEXT_ANSWER.lower() in answer.lower():
            return {**state, "answer": answer, "sources": [],"answerable": False,}
        pages = sorted({h["page"] for h in hits if h.get("page") is not None})
        return {**state, "answer": answer, "sources": pages,"answerable": True,}

    workflow = StateGraph(GraphState)
    workflow.add_node("understand_question", understand_question)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate_answer", generate_answer)

    workflow.set_entry_point("understand_question")
    workflow.add_edge("understand_question", "retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()
