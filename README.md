# AI Document Assistant — Conversational RAG with LangGraph

## Overview

Upload a PDF, get an automatic structured summary, and chat with the document
using Retrieval-Augmented Generation (RAG). Answers are grounded in the
uploaded document and include source pages.

## Features

- PDF validation and page-wise text extraction
- Recursive chunking with page metadata
- Local `intfloat/e5-base-v2` embeddings
- ChromaDB persistent vector storage with HNSW retrieval
- CrossEncoder reranking
- Automatic structured document summary
- Map-Reduce summarization for long documents
- Multi-turn conversational RAG with LangGraph
- Follow-up question rewriting
- Source page references
- Document hash-based duplicate detection
- Streamlit UI
- Optional FastAPI API
- Docker support
- Pytest tests and evaluation

## Full Project Flow

```text
                         USER
                           |
                           v
                    +-------------+
                    | Streamlit UI|
                    +------+------+
                           |
                           v
                     Upload PDF
                           |
                           v
                PDF Loader + Validation
                           |
                           v
                  Extract Document Text
                           |
              +------------+-------------+
              |                          |
              v                          v
        SUMMARY PATH                 RAG PATH
              |                          |
              v                          v
       Full Document Text             Chunking
              |                          |
       +------+-------+                  v
       |              |             E5 Embedding
     Short           Long                |
       |              |                  v
       v              v              ChromaDB
    Groq LLM      Map-Reduce             |
       |              |                  v
       +------+-------+             HNSW Retrieval
              |                          |
              v                          v
      Structured Summary          Candidate Chunks
                                         |
                                         v
                                  CrossEncoder
                                         |
                                         v
                                       Top-K
                                         |
                                         v
                                     LangGraph
                                         |
                                         v
                                      Groq LLM
                                         |
                                         v
                                  Answer + Sources
```

## LangGraph Workflow

```text
START
  |
  v
Understand / Rewrite Question
  |
  v
Retrieve from ChromaDB
  |
  v
CrossEncoder Reranking
  |
  v
Generate Answer
  |
  v
END
```

The LangGraph workflow is intentionally linear with three main nodes:
`understand_question`, `retrieve`, and `generate_answer`.

## Document Summarization

Short documents are summarized directly by the Groq LLM.

Long documents use Map-Reduce:

1. Summarize each chunk.
2. Combine partial summaries.
3. Generate the final structured summary.

The summary includes:

- Overview
- Main Topic
- Key Points
- Important Findings
- Conclusion

## Retrieval

The retrieval pipeline is:

```text
Query → E5 Embedding → ChromaDB HNSW → Candidate Chunks
      → CrossEncoder → Top-K → Groq LLM → Answer
```

The CrossEncoder reranks the candidates returned by ChromaDB and selects the
most relevant chunks for the final LLM context.

## Technologies

| Technology | Purpose |
|---|---|
| Streamlit | Web UI |
| pypdf | PDF extraction |
| LangChain | Chunking/integrations |
| LangGraph | RAG workflow |
| `intfloat/e5-base-v2` | Embeddings |
| ChromaDB + HNSW | Vector retrieval |
| CrossEncoder | Reranking |
| Groq | LLM |
| FastAPI | Optional API |
| Docker | Deployment |
| pytest | Testing |

## Configuration

Create `.env`:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
MODEL_NAME=llama-3.1-8b-instant

EMBEDDING_MODEL=intfloat/e5-base-v2

CHUNK_SIZE=1000
CHUNK_OVERLAP=150

TOP_K=4
RETRIEVAL_CANDIDATES=15

RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

CHROMA_PERSIST_DIR=chroma_db
DATA_DIR=data
MAX_UPLOAD_MB=25
```

Never commit the real `.env` file or API key.

## Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

Optional API:

```bash
uvicorn api:app --reload
```

## Docker

```bash
docker compose build
docker compose up
```

Open:

```text
http://localhost:8501
```

Check:

```bash
docker compose ps
```

Stop:

```bash
docker compose down
```

## Testing

```bash
pytest -v
```

## Evaluation

```bash
python evaluation/evaluate.py sample_data/sample_document.pdf
```

Results are written to:

```text
evaluation/results.json
```

## Project Structure

```text
ai-document-assistant/
├── evaluation/
├── sample_data/
├── src/
│   ├── chatbot.py
│   ├── chunking.py
│   ├── config.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── graph.py
│   ├── llm.py
│   ├── prompts.py
│   ├── retriever.py
│   ├── state.py
│   ├── summarizer.py
│   └── vector_store.py
├── tests/
├── app.py
├── api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Limitations

- OCR for scanned PDFs is not currently implemented.
- One active document session is supported at a time.
- Long-document summarization requires multiple LLM calls.
- Retrieval quality directly affects answer quality.
- CrossEncoder reranking adds some latency.

## Future Improvements

- OCR support
- DOCX support
- Multi-document workspaces
- Hybrid retrieval
- Table/image extraction
- Streaming responses
- Authentication
- Cloud deployment

## Author

**Prakash Kumar Shah**

GitHub: https://github.com/prakash22-26
