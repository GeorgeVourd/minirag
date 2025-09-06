# Mini RAG-based Question Answering API (LangChain + FAISS)

A minimal Retrieval-Augmented Generation (RAG) service built with **FastAPI**, **LangChain**, **LangGraph**, **FAISS**, and **OpenAI** (with **HuggingFace** fallback).  
Upload `.txt`/`.md` documents, ask questions, and get answers with cited sources.

## Features
- Ingest `.txt`/`.md`, split into chunks, and index into **FAISS**
- **/upload** endpoint for dynamic document updates
- **/ask** endpoint for question answering with sources
- Uses **OpenAI** for embeddings + LLM (or **HuggingFace** fallback if no key)
- Basic logging of Q/A and sources
- Index persistence to disk

## Requirements
- Python **3.11**

## Clone Repo
```PowerShell
cd "C:\Users\**USERNAME**\Documents"
git clone https://github.com/GeorgeVourd/minirag.git minirag_fresh
cd .\minirag_fresh
```
## Set venv and install requirements 
```PowerShell 
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
## Create .env from example
```PowerShell
Copy-Item .env.example .env
```
## Run
```Powershell
uvicorn app.main:app --reload
```
- Swagger UI: `http://localhost:8000/docs`
- Health check: `GET /health`

## Endpoints

### `POST /upload`
Upload and index a new document.
- **Form-Data**: `file=@/path/to/doc.md`

Example:
```bash
curl -X POST -F "file=@./sample.md" http://localhost:8000/upload
```

Response:
```json
{ "message": "Document 'sample.md' indexed successfully.", "chunks_indexed": 5 }
```

### `POST /ask`
Ask a question and get an answer grounded on your uploaded docs.

Request body:
```json
{ "question": "What is our return policy?" }
```

Ask question from .env setting:
```bash
curl -X POST -H "Content-Type: application/json"   -d '{"question":"What is our return policy?"}'   http://localhost:8000/ask
```

Ask Langchain
```bash
curl -X POST 'http://127.0.0.1:8000/ask?engine=chain'   -H 'Content-Type: application/json'   -d '{"question":"What is our return policy?"}'
```

Ask Langgraph
```bash
curl -X POST 'http://127.0.0.1:8000/ask?engine=graph'   -H 'Content-Type: application/json'   -d '{"question":"What is our return policy?"}'
```

Response (example):
```json
{
  "answer": "Customers may return products within 30 days for a full refund.",
  "sources": ["sample.md"]
}
```

## How it works (high level)
1. **Ingestion**: Files are split into overlapping chunks and embedded.
2. **Indexing**: Embeddings are stored in a **FAISS** index on disk (`data/index`).
3. **Retrieval**: For each question, the top-`k` chunks are retrieved by similarity.
4. **Generation**: An LLM (OpenAI or local HF) answers using only the retrieved context.

## Persistence
The FAISS index is saved under `data/index`. To reset the knowledge base, delete this folder.

## Logging
Logs are written to `logs/app.log` (and console). They include the question, the final answer, and the cited sources.

## HuggingFace Fallback Notes
- When `OPENAI_API_KEY` is **not** set, the app falls back to:
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
  - LLM: `google/flan-t5-base` (text2text generation)
- You may need to install **torch** for your platform; see PyTorch site for the correct wheel.

## Troubleshooting
- **No docs uploaded**: `/ask` returns 400 until you upload at least one document.
- **Index load issues**: Delete `data/index` and re-upload to rebuild.
- **HF model download**: First run may download models; ensure internet access or pre-download.

---

**Project Layout**
```
minirag/
├── app/
│   ├── main.py          # FastAPI app (routes: /upload, /ask, /ask_graph)
│   ├── ingestion.py     # load/split docs, build/load FAISS
│   ├── qa.py            # classic RAG (retriever + HF LLM) for /ask
│   ├── qa_graph.py      # LangGraph graph: state, nodes (retrieve→generate), edges, compile
│   └── utils.py         # config/env/logging helpers (incl. HF models, paths)
├── data/                # (runtime) sources & FAISS index
├── logs/                # (runtime) app logs
├── requirements.txt     # required python libraries
└── .env.example         # example of .env

```
