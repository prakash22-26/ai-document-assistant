"""
Optional FastAPI layer, per spec section 25.

Streamlit's own app.py calls src/chatbot.py directly and does not require
this file to run — this is provided so the same logic can be driven over
HTTP (e.g. from a non-Streamlit frontend) without duplicating logic.

Run with: uvicorn api:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.chatbot import Assistant
from src.document_loader import DocumentLoadError
from src.llm import LLMConfigError, friendly_error_message

app = FastAPI(title="AI Document Assistant API")
assistant = Assistant()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[int]


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    summary: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    file_bytes = await file.read()
    try:
        session = assistant.process_document(file_bytes, file.filename)
    except DocumentLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - catch-all so users never see a raw traceback
        raise HTTPException(
            status_code=500,
            detail=friendly_error_message(exc),
        ) from exc

    return UploadResponse(
        document_id=session.document_id,
        filename=session.filename,
        summary=session.summary,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = assistant.ask(request.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - catch-all so users never see a raw traceback
        raise HTTPException(
            status_code=500, detail=friendly_error_message(exc)
        ) from exc

    return ChatResponse(answer=result["answer"], sources=result["sources"])
