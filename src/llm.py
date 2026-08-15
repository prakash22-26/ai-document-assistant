"""
LLM provider abstraction.

The rest of the app calls get_llm() and never imports a provider SDK
directly. Supports Groq, OpenAI, Gemini, and Hugging Face.
"""

from __future__ import annotations

import re
import time

from src.config import settings


class LLMConfigError(Exception):
    """Raised when the configured LLM provider is missing required config."""


def get_llm():
    """
    Return a LangChain-compatible chat model based on LLM_PROVIDER.
    """

    provider = settings.llm_provider.lower().strip()

    # ---------------------------------------------------------
    # GROQ
    # ---------------------------------------------------------
    if provider == "groq":

        if not settings.groq_api_key:
            raise LLMConfigError(
                "GROQ_API_KEY is not set. Add it to your .env file. "
                "Get a free key at https://console.groq.com/keys"
            )

        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.model_name,
            api_key=settings.groq_api_key,
            temperature=0.1,
        )

    # ---------------------------------------------------------
    # OPENAI
    # ---------------------------------------------------------
    if provider == "openai":

        if not settings.openai_api_key:
            raise LLMConfigError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.model_name,
            api_key=settings.openai_api_key,
            temperature=0.1,
        )

    # ---------------------------------------------------------
    # GEMINI
    # ---------------------------------------------------------
    if provider == "gemini":

        if not settings.gemini_api_key:
            raise LLMConfigError(
                "GEMINI_API_KEY is not set. Add it to your .env file. "
                "Get a free key at https://aistudio.google.com/apikey"
            )

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.model_name,
            google_api_key=settings.gemini_api_key,
            temperature=0.1,
        )

    # ---------------------------------------------------------
    # HUGGING FACE
    # ---------------------------------------------------------
    if provider == "huggingface":

        if not settings.hf_api_key:
            raise LLMConfigError(
                "HF_API_KEY is not set. Add it to your .env file. "
                "Get a free token at https://huggingface.co/settings/tokens"
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.model_name,
            api_key=settings.hf_api_key,
            base_url="https://router.huggingface.co/v1",
            temperature=0.1,
        )

    raise LLMConfigError(
        f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. "
        "Supported: groq, openai, gemini, huggingface."
    )


# =============================================================
# RATE LIMIT HELPERS
# =============================================================

def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Detect common rate-limit errors across providers.
    """

    message = str(exc).lower()

    return any(
        value in message
        for value in [
            "429",
            "rate_limit_exceeded",
            "rate limit",
            "too many requests",
            "tokens per minute",
            "tpm",
        ]
    )


def _extract_wait_seconds(message: str) -> float | None:
    """
    Extract the provider's suggested retry time.

    Examples supported:

        try again in 30s
        try again in 2m
        try again in 2m43.728s
    """

    # ---------------------------------------------------------
    # 2m43.728s
    # ---------------------------------------------------------

    match = re.search(
        r"try again in\s+"
        r"(\d+(?:\.\d+)?)m"
        r"(\d+(?:\.\d+)?)s",
        message,
        re.IGNORECASE,
    )

    if match:

        minutes = float(match.group(1))
        seconds = float(match.group(2))

        return minutes * 60 + seconds

    # ---------------------------------------------------------
    # 30s
    # ---------------------------------------------------------

    match = re.search(
        r"try again in\s+"
        r"(\d+(?:\.\d+)?)s",
        message,
        re.IGNORECASE,
    )

    if match:
        return float(match.group(1))

    # ---------------------------------------------------------
    # 2m
    # ---------------------------------------------------------

    match = re.search(
        r"try again in\s+"
        r"(\d+(?:\.\d+)?)m",
        message,
        re.IGNORECASE,
    )

    if match:
        return float(match.group(1)) * 60

    return None


# =============================================================
# LLM INVOCATION WITH RETRY
# =============================================================

def invoke_with_retry(
    llm,
    prompt: str,
    max_retries: int = 1,
    base_delay: float = 2.0,
):
    """
    Invoke the LLM with rate-limit-aware retry handling.

    Important:
    - 429 errors are retried.
    - Provider-suggested wait time is respected.
    - Non-rate-limit errors are NOT retried.
    - 413/request-too-large errors are NOT retried.
    """

    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):

        try:
            return llm.invoke(prompt)

        except Exception as exc:

            last_exception = exc

            # -------------------------------------------------
            # Do not retry normal errors
            # -------------------------------------------------

            if not _is_rate_limit_error(exc):
                raise

            # -------------------------------------------------
            # No retries remaining
            # -------------------------------------------------

            if attempt >= max_retries:
                raise

            # -------------------------------------------------
            # Read provider's suggested wait time
            # -------------------------------------------------

            wait_seconds = _extract_wait_seconds(
                str(exc)
            )

            # -------------------------------------------------
            # If provider did not provide a wait time,
            # use exponential backoff.
            # -------------------------------------------------

            if wait_seconds is None:

                wait_seconds = (
                    base_delay * (2 ** attempt)
                )

            print(
                f"[llm] Rate limit reached. "
                f"Waiting {wait_seconds:.1f} seconds "
                f"before retry "
                f"{attempt + 1}/{max_retries}..."
            )
            if wait_seconds > 30:
                raise
            time.sleep(wait_seconds)

    if last_exception:
        raise last_exception

    raise RuntimeError(
        "LLM invocation failed."
    )


# =============================================================
# USER-FRIENDLY ERROR
# =============================================================

def friendly_error_message(exc: Exception) -> str:
    """
    Convert provider errors into user-friendly messages.
    """

    message = str(exc)

    if _is_rate_limit_error(exc):

        wait_seconds = _extract_wait_seconds(
            message
        )

        if wait_seconds is not None:

            minutes = int(
                wait_seconds // 60
            )

            seconds = int(
                wait_seconds % 60
            )

            if minutes > 0:

                return (
                    "The AI provider's rate limit "
                    f"was reached. Please wait "
                    f"about {minutes}m {seconds}s "
                    "and try again."
                )

            return (
                "The AI provider's rate limit "
                f"was reached. Please wait "
                f"about {seconds} seconds "
                "and try again."
            )

        return (
            "The AI provider's rate limit was "
            "reached. Please wait a few minutes "
            "and try again, or switch to another "
            "model/provider."
        )

    if (
        "402" in message
        or (
            "credit" in message.lower()
            and "depleted" in message.lower()
        )
    ):

        return (
            "The AI provider's free usage credits "
            "are exhausted. Try a different "
            "provider or model."
        )

    return (
        "Something went wrong talking to the "
        "AI provider. Please try again."
    )
