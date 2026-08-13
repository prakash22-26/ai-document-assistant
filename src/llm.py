"""
LLM provider abstraction.

The rest of the app calls `get_llm()` and never imports a provider SDK
directly, so swapping LLM_PROVIDER in .env doesn't touch other modules.
Currently supports Groq (OpenAI-compatible chat API) since it has a
generous free tier for local development, per the spec. Add a new
provider by adding a branch in get_llm().
"""
from __future__ import annotations

from src.config import settings


class LLMConfigError(Exception):
    """Raised when the configured LLM provider is missing required config."""


def get_llm():
    """
    Returns a LangChain-compatible chat model instance based on
    settings.llm_provider / settings.model_name / settings.groq_api_key.
    """
    provider = settings.llm_provider.lower()
 
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

    if provider == "huggingface":
        if not settings.hf_api_key:
            raise LLMConfigError(
                "HF_API_KEY is not set. Add it to your .env file. "
                "Get a free token at https://huggingface.co/settings/tokens"
            )
        from langchain_openai import ChatOpenAI
 
        # Hugging Face's Inference Providers router is OpenAI-compatible,
        # so this reuses the same client as the openai branch, just
        # pointed at a different base URL.
        return ChatOpenAI(
            model=settings.model_name,
            api_key=settings.hf_api_key,
            base_url="https://router.huggingface.co/v1",
            temperature=0.1,
        )
 
    raise LLMConfigError(
        f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. "
        "Supported: groq, openai, gemini. Add another branch in "
        "src/llm.py to support more providers."
    )

def invoke_with_retry(llm, prompt: str, max_retries: int = 3, base_delay: float = 2.0):
    """Invoke the LLM with basic exponential backoff on transient rate-limit errors.

    Groq's free tier has a low tokens-per-minute budget (e.g. 6000 TPM on
    llama-3.1-8b-instant), which a short burst of calls can hit even on a
    small document. This retries transient 429s only. It does NOT retry a
    413 "request too large" error, since that will never succeed no matter
    how many times you retry — the caller must reduce the request size
    instead (see summarizer.py's batching).
    """
    import time

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            message = str(exc)
            if "429" in message or "rate_limit_exceeded" in message.lower():
                last_exc = exc
                time.sleep(base_delay * (attempt + 1))
                continue
            raise
    raise last_exc

def friendly_error_message(exc: Exception) -> str:
    """Turn a raw exception into a message safe and useful to show a user.
 
    Detects rate-limit errors specifically (works across providers, since
    it matches on error text rather than a provider-specific exception
    class) and extracts the provider's own suggested wait time when
    present, so the user knows to wait rather than assume the app is
    broken. Falls back to a generic message for anything else — never
    shows a raw stack trace or internal detail to the user.
    """
    import re
 
    message = str(exc)
 
    if "429" in message or "rate_limit" in message.lower():
        wait_match = re.search(r"try again in ([\d.]+[a-z]+(?:[\d.]+[a-z]+)*)", message, re.IGNORECASE)
        if wait_match:
            return (
                f"The AI provider's rate limit was reached. Please wait "
                f"about {wait_match.group(1)} and try again."
            )
        return (
            "The AI provider's rate limit was reached. Please wait a few "
            "minutes and try again, or switch to a different model/provider "
            "in your .env file."
        )
 
    if "402" in message or "credit" in message.lower() and "depleted" in message.lower():
        return (
            "The AI provider's free usage credits are exhausted for this "
            "period. Try a different provider or model in your .env file."
        )
 
    return "Something went wrong talking to the AI provider. Please try again."