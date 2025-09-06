# Mini RAG-based Question Answering API (LangChain + FAISS)

A minimal Retrieval-Augmented Generation (RAG) service built with **FastAPI**, **LangChain**, **FAISS**, and **OpenAI** (with **HuggingFace** fallback).  
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

## Install
```bash
# from project root
pip install -r requirements.txt
```
> For local HuggingFace LLM fallback, you may also need a compatible **PyTorch** build. See https://pytorch.org/

## Configure
Create a `.env` in the project root (same folder as `requirements.txt`) by copying `.env.example`:
```bash
cp .env.example .env
```
Fill in your `OPENAI_API_KEY` if you want to use OpenAI. If you leave it empty, the app will use the HuggingFace fallback models defined in `.env`.

## Run
```bash
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

Example:
```bash
curl -X POST -H "Content-Type: application/json"   -d '{"question":"What is our return policy?"}'   http://localhost:8000/ask


**Bonus***
# For langchain execution no matter the choice on .env
curl -X POST 'http://127.0.0.1:8000/ask?engine=chain'   -H 'Content-Type: application/json'   -d '{"question":"What is our return policy?"}'

# For langgraph execution regardless of .env setting

curl -X POST 'http://127.0.0.1:8000/ask?engine=graph'   -H 'Content-Type: application/json'   -d '{"question":"What is our return policy?"}'
```

Response (example):
```json
{
  "answer": "Customers may return products within 30 days for a full refund.",
  "sources": ["policies.md"]
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
- Local models are slower and less capable than OpenAI GPT models; use for demo/offline mode.

## Troubleshooting
- **No docs uploaded**: `/ask` returns 400 until you upload at least one document.
- **Index load issues**: Delete `data/index` and re-upload to rebuild.
- **HF model download**: First run may download models; ensure internet access or pre-download.

---

**Project Layout**
```
mini_rag_qa/
├── app/
│   ├── main.py          # FastAPI app (routes)
│   ├── ingestion.py     # load/split docs, FAISS save/load
│   ├── qa.py            # embeddings, LLM, retrieval chain
│   └── utils.py         # config/env/logging helpers
├── data/                # (created at runtime) initial docs or saved index
├── logs/                # (created at runtime) app logs
├── requirements.txt
└── .env.example
```
