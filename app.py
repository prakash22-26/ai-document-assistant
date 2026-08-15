"""Streamlit UI for the AI Document Assistant."""

from __future__ import annotations

import hashlib

import streamlit as st

from src.chatbot import Assistant
from src.document_loader import DocumentLoadError
from src.llm import LLMConfigError, friendly_error_message


st.set_page_config(
    page_title="AI Document Assistant",
    page_icon=None,
    layout="centered",
)


def get_assistant() -> Assistant:
    """
    Create one Assistant for the current Streamlit session.

    Do not use @st.cache_resource here because it can keep
    the Assistant outside the current browser session.
    """

    if "assistant" not in st.session_state:
        st.session_state.assistant = Assistant()

    return st.session_state.assistant


def reset_document():
    """Reset the active document and chat session."""

    st.session_state.assistant = Assistant()
    st.session_state.processed_hash = None
    st.session_state.messages = []


def get_file_hash(file_bytes: bytes) -> str:
    """Return a stable SHA-256 hash for the uploaded PDF."""

    return hashlib.sha256(file_bytes).hexdigest()


def main():

    st.title("🤖 AI Document Assistant")

    st.caption(
        "Upload a PDF, get an instant summary, then chat with it — "
        "answers are grounded in the document."
    )

    # ---------------------------------------------------------
    # Initialize session state
    # ---------------------------------------------------------

    if "processed_hash" not in st.session_state:
        st.session_state.processed_hash = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    assistant = get_assistant()

    # ---------------------------------------------------------
    # PDF upload
    # ---------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf"],
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        current_hash = get_file_hash(file_bytes)

        # -----------------------------------------------------
        # New PDF uploaded
        # -----------------------------------------------------

        if (
            st.session_state.processed_hash is not None
            and current_hash
            != st.session_state.processed_hash
        ):

            reset_document()

            assistant = st.session_state.assistant

        # -----------------------------------------------------
        # Process PDF
        # -----------------------------------------------------

        if (
            st.session_state.processed_hash
            != current_hash
        ):

            with st.spinner(
                "Processing..."
            ):

                try:

                    assistant.process_document(
                        file_bytes,
                        uploaded_file.name,
                    )

                    st.session_state.processed_hash = (
                        current_hash
                    )

                    # New document = new chat
                    st.session_state.messages = []

                except DocumentLoadError as exc:

                    st.error(str(exc))
                    return

                except LLMConfigError as exc:

                    st.error(str(exc))
                    return

                except Exception as exc:

                    st.error(
                        friendly_error_message(exc)
                    )

                    print(
                        f"[upload error] {exc!r}"
                    )

                    return

    # ---------------------------------------------------------
    # Show document
    # ---------------------------------------------------------

    if assistant.session is not None:

        st.subheader("📋 Document Summary")

        st.markdown(
            assistant.session.summary
        )

        st.divider()

        st.subheader(
            "💬 Chat with Document"
        )

        # -----------------------------------------------------
        # Chat history
        # -----------------------------------------------------

        for msg in st.session_state.messages:

            with st.chat_message(
                msg["role"]
            ):

                st.markdown(
                    msg["content"]
                )

                if msg.get("sources"):

                    st.caption(
                        "Sources: "
                        + ", ".join(
                            f"Page {p}"
                            for p in msg["sources"]
                        )
                    )

        # -----------------------------------------------------
        # User question
        # -----------------------------------------------------

        question = st.chat_input(
            "Ask a question..."
        )

        if question:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):

                st.markdown(question)

            # -------------------------------------------------
            # Generate answer
            # -------------------------------------------------

            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    try:

                        result = assistant.ask(
                            question
                        )

                    except LLMConfigError as exc:

                        st.error(str(exc))
                        return

                    except Exception as exc:

                        st.error(
                            friendly_error_message(
                                exc
                            )
                        )

                        print(
                            f"[chat error] {exc!r}"
                        )

                        return

                st.markdown(
                    result["answer"]
                )

                if result.get("sources"):

                    st.caption(
                        "Sources: "
                        + ", ".join(
                            f"Page {p}"
                            for p in result["sources"]
                        )
                    )

            # -------------------------------------------------
            # Save assistant response
            # -------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get(
                        "sources",
                        [],
                    ),
                }
            )

    else:

        st.info(
            "Upload a PDF above to get started."
        )


if __name__ == "__main__":
    main()
