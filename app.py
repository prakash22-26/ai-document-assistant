"""Streamlit UI for the AI Document Assistant."""
from __future__ import annotations

import streamlit as st

from src.chatbot import Assistant
from src.document_loader import DocumentLoadError
from src.llm import LLMConfigError, friendly_error_message

st.set_page_config(page_title="AI Document Assistant", page_icon="", layout="centered")


@st.cache_resource
def get_assistant() -> Assistant:
    return Assistant()


def main():
    st.title("🤖 AI Document Assistant")
    st.caption("Upload a PDF, get an instant summary, then chat with it — answers are grounded in the document.")

    assistant = get_assistant()

    if "processed_hash" not in st.session_state:
        st.session_state.processed_hash = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    uploaded_file = st.file_uploader("Upload Document", type=["pdf"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()

        if st.session_state.processed_hash != hash(file_bytes):
            with st.spinner("Extracting text, chunking, embedding, and summarizing..."):
                try:
                    assistant.process_document(file_bytes, uploaded_file.name)
                    st.session_state.processed_hash = hash(file_bytes)
                    st.session_state.messages = []  # new doc -> clear old chat context
                except DocumentLoadError as exc:
                    st.error(str(exc))
                    return
                except LLMConfigError as exc:
                    st.error(str(exc))
                    return
                except Exception as exc:  # pragma: no cover - defensive UI guard
                    st.error(friendly_error_message(exc))
                    print(f"[upload error] {exc!r}")  # logged server-side only, not shown to the user
                    return

    if assistant.session is not None:
        st.subheader("📋 Document Summary")
        st.markdown(assistant.session.summary)

        st.divider()
        st.subheader("💬 Chat with Document")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    st.caption("Sources: " + ", ".join(f"Page {p}" for p in msg["sources"]))

        question = st.chat_input("Ask a question...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = assistant.ask(question)
                    except LLMConfigError as exc:
                        st.error(str(exc))
                        return
                    except Exception as exc:  # pragma: no cover
                        st.error(friendly_error_message(exc))
                        print(f"[chat error] {exc!r}")  # logged server-side only, not shown to the user
                        return
                st.markdown(result["answer"])
                if result["sources"]:
                    st.caption("Sources: " + ", ".join(f"Page {p}" for p in result["sources"]))

            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            )
    else:
        st.info("Upload a PDF above to get started.")


if __name__ == "__main__":
    main()
