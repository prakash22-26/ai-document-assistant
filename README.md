#  AI Document Assistant

### Conversational RAG for Intelligent PDF Understanding

[![Live Demo](https://img.shields.io/badge/Live-Demo-FF4B4B?logo=streamlit&logoColor=white)](https://ai-document-assistant-g7edfcdfs2khntuzquhsuh.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/prakash22-26/ai-document-assistant)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-1C3C3C)](https://www.langchain.com/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-FF6F61)](https://www.trychroma.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Groq](https://img.shields.io/badge/Groq-LLM-F55036)](https://groq.com/)

> **Upload a PDF → Summarize → Ask Questions → Retrieve → Rerank → Get Grounded Answers**

### 🚀 [Try the Live Demo](https://ai-document-assistant-g7edfcdfs2khntuzquhsuh.streamlit.app/)

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [📝 Document Summarization](#-document-summarization)
- [💬 Conversational Memory](#-conversational-memory)
- [🔍 Retrieval & Reranking](#-retrieval--reranking)
- [🔗 LangGraph Workflow](#-langgraph-workflow)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [🔧 Configuration](#-configuration)
- [🧪 Testing](#-testing)
- [📊 Evaluation](#-evaluation)
- [🌐 FastAPI API](#-fastapi-api)
- [🎯 Use Cases](#-use-cases)
- [⚠️ Limitations](#️-limitations)
- [🚀 Future Improvements](#-future-improvements)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👨‍💻 Author](#-author)

---

# 🌟 Overview

**AI Document Assistant** is a conversational **Retrieval-Augmented Generation
(RAG)** application for interacting with uploaded PDF documents.

The application automatically generates a structured summary of the uploaded
document and allows users to ask questions about its contents.

Instead of sending the entire document to the LLM for every question, the
system retrieves relevant chunks from the document, reranks them using a
CrossEncoder, and provides the most relevant context to the LLM.

The project uses:

- 📄 PDF text extraction
- ✂️ Recursive document chunking
- 🧠 Local Hugging Face embeddings
- 🗄️ ChromaDB vector storage
- 🔎 HNSW semantic retrieval
- 🎯 CrossEncoder reranking
- 🔗 LangGraph orchestration
- 💬 Session-based conversational memory
- 📝 Automatic document summarization
- 🤖 Groq LLM inference
- 📚 Source page references

No LLM is fine-tuned on the uploaded documents.

---

# ✨ Features

## 📄 Document Processing

- PDF upload and validation
- Page-wise text extraction using `pypdf`
- Recursive text chunking
- Page metadata preservation
- Document content hashing
- Duplicate document detection
- Persistent ChromaDB storage

## 📝 Automatic Document Summarization

- Automatic summary after PDF upload
- Structured document summary
- Short-document direct summarization
- Long-document Map-Reduce summarization
- Chunk-level partial summaries
- Final combined summary

The generated summary focuses on:

- Overview
- Main Topic
- Key Points
- Important Findings
- Conclusion

## 💬 Conversational Q&A

- Multi-turn document conversations
- Session-based `chat_history`
- Context-aware follow-up questions
- Follow-up question rewriting
- Grounded document answers
- Source page references
- Explicit no-context response

## 🔍 Retrieval

- Local E5 embeddings
- ChromaDB persistent vector storage
- HNSW approximate nearest-neighbor retrieval
- Configurable candidate retrieval
- Document-specific retrieval

## 🎯 Reranking

The application performs a second-stage relevance check using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Pipeline:

```text
Query
  ↓
Dense Retrieval
  ↓
Candidate Chunks
  ↓
CrossEncoder
  ↓
Sorted Relevant Chunks
  ↓
Top-K
```

## 🔗 LangGraph

The conversational RAG workflow is implemented using LangGraph.

```text
START
  ↓
Understand / Rewrite Question
  ↓
Retrieve
  ↓
Generate Answer
  ↓
END
```

## 🌐 Optional FastAPI

A FastAPI layer is included for accessing the application through HTTP
endpoints separately from the Streamlit UI.

## 🧪 Testing & Evaluation

- Pytest test suite
- Retrieval evaluation
- Groundedness/no-context evaluation
- Answer correctness heuristic
- Evaluation results stored in `evaluation/results.json`

---

# 🏗️ Architecture

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
                    PDF Validation + pypdf
                                |
                                v
                       Extract Page Text
                                |
                    +-----------+-----------+
                    |                       |
                    v                       v
             SUMMARY PATH               RAG PATH
                    |                       |
                    v                       v
           Full Document Text            Chunking
                    |                       |
             +------+-------+               v
             |              |         E5 Embeddings
           Short           Long             |
             |              |               v
             v              v           ChromaDB
         Direct         Map-Reduce          |
         Summary         Summary             v
             |              |          HNSW Retrieval
             +------+-------+               |
                    |                       v
                    v                Candidate Chunks
             Structured Summary             |
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
                                           LLM
                                            |
                                            v
                                     Answer + Sources
                                            |
                                            v
                                    Update Chat History
                                            |
                                            +----> Next Question
```

### Summary Path

```text
PDF
 ↓
Full Extracted Text
 ↓
Short / Long Decision
 ↓
Direct LLM OR Map-Reduce
 ↓
Structured Summary
```

### Question Answering Path

```text
Question
 ↓
Query Embedding
 ↓
ChromaDB HNSW
 ↓
Candidate Chunks
 ↓
CrossEncoder
 ↓
Top-K
 ↓
LangGraph
 ↓
LLM
 ↓
Answer + Sources
```

---

# 📝 Document Summarization

Summarization is independent of vector retrieval.

The application does **not** create the document summary from retrieved
chunks.

## Short Documents

```text
Full Document Text
        ↓
Summary Prompt
        ↓
LLM
        ↓
Structured Summary
```

## Long Documents

```text
Full Document
      ↓
Document Chunks
      ↓
Summarize Each Chunk
      ↓
Partial Summaries
      ↓
Combine Partial Summaries
      ↓
Final LLM Reduction
      ↓
Structured Summary
```

The current implementation uses approximately:

```text
LONG_DOCUMENT_CHAR_THRESHOLD = 3000
```

and batches large groups of partial summaries using a reduction character
limit.

---

# 💬 Conversational Memory

The application uses **session-based conversational memory**.

Memory is stored in the active `DocumentSession`:

```python
chat_history: list[dict]
```

The session contains:

```text
document_id
filename
document_hash
summary
chat_history
```

After a successful answer, the application adds the user question and
assistant answer to the history.

### Example

```text
User:
What is the main topic?

Assistant:
The document discusses ...

User:
What are its main benefits?
```

The system uses the previous conversation to understand what "its" refers to.

```text
Chat History + Follow-up Question
              ↓
      Question Rewriting
              ↓
       Standalone Question
              ↓
          Retrieval
```

This is **short-term/session memory**, not persistent long-term user memory.

When a new document session is created, the active conversation history is
reset.

---

# 🔍 Retrieval & Reranking

The project uses a two-stage retrieval architecture.

## Stage 1 — Dense Retrieval

The default embedding model is:

```text
intfloat/e5-base-v2
```

```text
User Question
      ↓
Query Embedding
      ↓
ChromaDB
      ↓
HNSW Search
      ↓
Candidate Chunks
```

The default candidate count is:

```env
RETRIEVAL_CANDIDATES=15
```

## Stage 2 — CrossEncoder Reranking

The candidates are converted into query-document pairs:

```text
(query, chunk_1)
(query, chunk_2)
(query, chunk_3)
...
```

The CrossEncoder:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

scores each pair and the candidates are sorted by relevance.

Example:

```env
TOP_K=7
```

Complete pipeline:

```text
Question
   ↓
E5 Embedding
   ↓
ChromaDB HNSW
   ↓
15 Candidate Chunks
   ↓
CrossEncoder
   ↓
Rank by Relevance
   ↓
Top 7
   ↓
LLM
   ↓
Answer
```

---

# 🔗 LangGraph Workflow

The conversational RAG pipeline uses a simple three-node LangGraph.

```text
START
  |
  v
+-------------------------+
| understand_question     |
|                         |
| Question + Chat History |
|          ↓              |
| Standalone Question     |
+------------+------------+
             |
             v
+-------------------------+
| retrieve                |
|                         |
| ChromaDB Retrieval      |
|          ↓              |
| CrossEncoder Reranking  |
|          ↓              |
| Top-K Documents         |
+------------+------------+
             |
             v
+-------------------------+
| generate_answer         |
|                         |
| Context + Question      |
|          ↓              |
| LLM                     |
|          ↓              |
| Answer + Sources        |
+------------+------------+
             |
             v
            END
```

The graph is intentionally linear and is **not a multi-agent system**.

The three nodes are:

```text
understand_question
retrieve
generate_answer
```

---

# 🤖 Answer Generation

The retrieved chunks are formatted with their source page information and
passed to the answer prompt.

```text
Chat History
     +
Retrieved Context
     +
Standalone Question
     ↓
    LLM
     ↓
Grounded Answer
     +
Source Pages
```

If no usable documents are retrieved, the application returns:

```text
I couldn't find this information in the uploaded document.
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core application language |
| **Streamlit** | Web UI |
| **pypdf** | PDF extraction |
| **LangChain** | Chunking and integrations |
| **LangGraph** | Conversational RAG workflow |
| **Hugging Face** | Local embedding/reranking models |
| **E5 Base v2** | Document/query embeddings |
| **ChromaDB** | Persistent vector database |
| **HNSW** | Approximate nearest-neighbor search |
| **CrossEncoder** | Candidate reranking |
| **Groq** | LLM inference |
| **FastAPI** | Optional REST API |
| **pytest** | Automated testing |
| **Git/GitHub** | Version control |

---

# 📁 Project Structure

```text
ai-document-assistant/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│
├── evaluation/
│   ├── evaluate.py
│   ├── generate_questions.py
│   ├── questions.json
│   └── results.json
│
├── sample_data/
│   ├── sample_document.pdf
│   ├── sample_document.txt
│   └── warehouse.pdf
│
├── src/
│   ├── __init__.py
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
│
├── tests/
│   ├── test_chunking.py
│   ├── test_document_loader.py
│   ├── test_graph.py
│   └── test_retrieval.py
│
├── api.py
├── app.py
├── pytest.ini
├── requirements.txt
└── README.md
```

---

# 🚀 Quick Start

## Prerequisites

- Python 3.10+
- Git
- API key for the configured LLM provider

## 1. Clone the Repository

```bash
git clone https://github.com/prakash22-26/ai-document-assistant.git
cd ai-document-assistant
```

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Create `.env`

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
MODEL_NAME=llama-3.1-8b-instant

EMBEDDING_MODEL=intfloat/e5-base-v2

CHUNK_SIZE=1000
CHUNK_OVERLAP=150

TOP_K=7
RETRIEVAL_CANDIDATES=15

RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

CHROMA_PERSIST_DIR=chroma_db
DATA_DIR=data
MAX_UPLOAD_MB=25
```

> Never commit your real `.env` file or API key.

## 5. Run

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# 🌐 Live Demo

The application is deployed on Streamlit:

### 🚀 [Open AI Document Assistant](https://ai-document-assistant-g7edfcdfs2khntuzquhsuh.streamlit.app/)

You can upload a supported PDF, view its generated summary, and ask
document-grounded questions.

---

# 🔧 Configuration

Configuration is centralized in:

```text
src/config.py
```

### LLM

```env
LLM_PROVIDER=groq
MODEL_NAME=llama-3.1-8b-instant
GROQ_API_KEY=your_api_key
```

### Embeddings

```env
EMBEDDING_MODEL=intfloat/e5-base-v2
```

### Chunking

```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

### Retrieval

```env
RETRIEVAL_CANDIDATES=15
TOP_K=7
```

### Reranker

```env
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Storage

```env
CHROMA_PERSIST_DIR=chroma_db
DATA_DIR=data
```

### Upload Limit

```env
MAX_UPLOAD_MB=25
```

---

# 📄 PDF Processing

The document loader validates the uploaded PDF before processing.

It handles cases such as:

- Empty files
- Invalid PDFs
- Corrupt PDFs
- PDFs with no pages
- PDFs with no extractable text
- Scanned/image-only PDFs

Text is extracted page-by-page using `pypdf`.

Page metadata is preserved in chunks so answers can include source page
references.

---

# 🔁 Duplicate Document Detection

The application calculates a content hash for uploaded documents.

Conceptually:

```text
PDF
 ↓
Content Hash
 ↓
Already Processed?
 ↓
YES
 ↓
Reuse Existing Document ID
 ↓
Skip Re-embedding
```

This prevents unnecessary embedding work when the same document is uploaded
again.

---

# 🗃️ Document Session

The application intentionally supports one active document session.

A session contains:

```text
document_id
filename
document_hash
summary
chat_history
```

Uploading another document replaces the active session and starts a new
conversation.

---

# 🧪 Testing

Run:

```bash
pytest -v
```

The test suite covers areas including:

- PDF extraction
- Invalid/corrupt PDF handling
- Chunking
- Metadata preservation
- Retrieval
- LangGraph execution
- No-context behavior
- Conversation history behavior

---

# 📊 Evaluation

Run:

```bash
python evaluation/evaluate.py sample_data/sample_document.pdf
```

Results are written to:

```text
evaluation/results.json
```

The evaluation checks:

- Retrieval relevance
- Grounded/no-context behavior
- Answer correctness

The answer-correctness component uses a simple keyword-overlap heuristic and
should be treated as a smoke test rather than a rigorous benchmark.

---

# 🌐 FastAPI API

The repository includes an optional FastAPI layer.

Start it with:

```bash
uvicorn api:app --reload
```

Typical endpoints:

```text
POST /documents/upload
POST /chat
GET  /health
```

---

# 🎯 Use Cases

### 🎓 Academic

- Research paper analysis
- Study material Q&A
- Thesis/document exploration
- Technical document understanding

### 💼 Business

- Business report analysis
- Meeting documentation
- Project documentation
- Requirements analysis

### 📚 Technical Documentation

- API documentation
- Software documentation
- System specifications
- Technical reports

### 📋 General PDF Analysis

- PDF summarization
- Document Q&A
- Information discovery
- Multi-turn conversations

---

# ⚠️ Limitations

- OCR for scanned PDFs is not currently implemented.
- Only one active document session is supported.
- Memory is session-based, not persistent long-term memory.
- Retrieval quality directly affects answer quality.
- CrossEncoder reranking adds inference latency.
- LLM calls are required for summarization and answer generation.
- Very long documents can require multiple summarization calls.

---

# 🚀 Future Improvements

- 🔤 OCR for scanned PDFs
- 📑 DOCX support
- 🗂️ Multi-document workspaces
- 🔀 Hybrid BM25 + dense retrieval
- 📊 Larger retrieval evaluation datasets
- 🧾 Table and image extraction
- ⚡ Streaming responses
- 🔐 Authentication
- ☁️ Cloud deployment improvements
- 🧠 Persistent user memory
- 📈 More rigorous RAG evaluation
- 🎯 Retrieval confidence thresholds

---

# 🎓 Learning Highlights

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation
- Sentence embeddings
- Vector databases
- HNSW retrieval
- CrossEncoder reranking
- Prompt engineering
- Conversational memory
- Query rewriting
- LangGraph workflows
- Map-Reduce summarization
- FastAPI
- Streamlit
- Unit testing
- RAG evaluation
- Git/GitHub workflows

---

# 🤝 Contributing

Contributions are welcome.

## Development Workflow

```bash
git checkout -b feature/your-feature
```

Run tests:

```bash
pytest -v
```

Commit:

```bash
git add .
git commit -m "Add your feature"
```

Push:

```bash
git push origin feature/your-feature
```

Then create a Pull Request.

### Guidelines

- Keep modules focused.
- Follow Python coding conventions.
- Add tests for new functionality.
- Use environment variables for secrets.
- Never commit API keys.
- Update documentation when architecture changes.

---

# 📄 License

No explicit `LICENSE` file is currently included in the repository.

If you intend to release the project as open source, add a `LICENSE` file and
specify the selected license here.

---

# 👨‍💻 Author

## Prakash Kumar Shah

Computer Science & Engineering

[![GitHub](https://img.shields.io/badge/GitHub-prakash22--26-181717?logo=github&logoColor=white)](https://github.com/prakash22-26)

### Repository

[⭐ AI Document Assistant](https://github.com/prakash22-26/ai-document-assistant)

### Live Application

[🚀 AI Document Assistant](https://ai-document-assistant-g7edfcdfs2khntuzquhsuh.streamlit.app/)

---

## 💬 Final Thoughts

AI Document Assistant demonstrates a complete conversational RAG pipeline:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Local Embeddings
 ↓
ChromaDB / HNSW
 ↓
Candidate Retrieval
 ↓
CrossEncoder Reranking
 ↓
LangGraph
 ↓
LLM
 ↓
Grounded Answer + Sources
 ↓
Session Chat History
```

At the same time, document summarization follows its own path:

```text
PDF Text
   ↓
Short / Long Decision
   ↓
Direct Summary OR Map-Reduce
   ↓
Structured Summary
```

This separation keeps document summarization independent from question
retrieval while allowing the RAG pipeline to focus on finding relevant context
for each question.

---

### ⭐ If you find this project useful, consider starring the repository.

**Built with Python • Streamlit • LangGraph • ChromaDB • Hugging Face • Groq**
