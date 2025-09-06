# main.py
# FastAPI app for a Mini RAG-based Question Answering API using LangChain + FAISS (+ optional LangGraph)
# Endpoints:
#   - GET  /            : hello
#   - GET  /health      : status + which engine default
#   - POST /upload      : dynamically add new documents (.txt, .md)
#   - POST /ask         : ask a question, returns {"answer": "...", "sources": ["file1.txt", ...]}
#
# Run:
#   python -m uvicorn app.main:app --reload

import os
from pathlib import Path
from typing import Optional, Set, Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

from .utils import get_logger, ensure_dirs, load_env, Config
from .ingestion import (
    ALLOWED_EXTS,
    load_file_to_documents,
    split_text_documents,
    load_vectorstore,
    save_vectorstore,
    create_vectorstore_from_docs,
)
from .qa import (
    get_embeddings,
    get_llm,
    build_retrieval_chain,
    answer_with_sources,
)

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------

load_env()
cfg = Config()

ensure_dirs([cfg.DATA_DIR, cfg.INDEX_DIR, cfg.LOG_DIR])

logger = get_logger("mini-rag", log_file=str(Path(cfg.LOG_DIR) / "app.log"))
app = FastAPI(title="Mini RAG QA API", version="1.2.0")

# Optional: LangGraph integration (loaded after logger so we can log failures)
LANGGRAPH_AVAILABLE = False
try:
    from .qa_graph import build_langgraph_chain, answer_with_langgraph  # type: ignore
    LANGGRAPH_AVAILABLE = True
    logger.info("LangGraph module loaded.")
except Exception as e:
    logger.info(f"LangGraph not available: {e}")

def _truthy(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "on"}

# Keep user preference in app.state (default ON; change with USE_LANGGRAPH=0)
app.state.use_langgraph = _truthy(os.getenv("USE_LANGGRAPH", "1"))

# Runtime state
app.state.embeddings = get_embeddings(cfg, logger)
app.state.llm = get_llm(cfg, logger)
app.state.vectorstore = load_vectorstore(cfg.INDEX_DIR, app.state.embeddings, logger)

# Two alternative pipelines:
# - retrieval_chain (LangChain RetrievalQA)
# - graph_chain (LangGraph)
app.state.retrieval_chain = None
app.state.graph_chain = None

if app.state.vectorstore is not None:
    # Always build the default retrieval chain
    app.state.retrieval_chain = build_retrieval_chain(app.state.llm, app.state.vectorstore, cfg.TOP_K, logger)
    # Build LangGraph chain if available
    if LANGGRAPH_AVAILABLE:
        app.state.graph_chain = build_langgraph_chain(app.state.llm, app.state.vectorstore, cfg.TOP_K, logger)
        logger.info("LangGraph chain initialized.")
    logger.info("Vectorstore loaded and pipelines initialized.")
else:
    logger.info("No existing vectorstore found. Upload documents via /upload to initialize.")

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "welcome to mini-rag!"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "langgraph": LANGGRAPH_AVAILABLE,
        "prefer_langgraph": bool(app.state.use_langgraph),
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Validate extension
    filename = file.filename or "uploaded_file"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTS)}")

    try:
        raw = await file.read()
        text = raw.decode("utf-8", errors="ignore")
        docs = load_file_to_documents(text=text, source=filename)
        chunks = split_text_documents(docs, chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)

        if app.state.vectorstore is None:
            # First time: create index
            app.state.vectorstore = create_vectorstore_from_docs(chunks, app.state.embeddings, logger)
        else:
            # Incremental update
            app.state.vectorstore.add_documents(chunks)

        save_vectorstore(app.state.vectorstore, cfg.INDEX_DIR, logger)

        # Rebuild pipelines
        app.state.retrieval_chain = build_retrieval_chain(app.state.llm, app.state.vectorstore, cfg.TOP_K, logger)
        if LANGGRAPH_AVAILABLE:
            app.state.graph_chain = build_langgraph_chain(app.state.llm, app.state.vectorstore, cfg.TOP_K, logger)

        logger.info(f"Indexed {len(chunks)} chunks from '{filename}'.")
        return {"message": f"Document '{filename}' indexed successfully.", "chunks_indexed": len(chunks)}
    except Exception as e:
        logger.exception("Failed to process upload")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.post("/ask")
async def ask(
    req: AskRequest,
    engine: Optional[Literal["graph", "chain"]] = Query(
        None,
        description="Override engine per-call: 'graph' (LangGraph) or 'chain' (RetrievalQA).",
    ),
):
    if app.state.vectorstore is None:
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Upload documents via /upload first.")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        # decide engine: per-call override > global default > fallback
        if engine == "graph":
            use_graph = LANGGRAPH_AVAILABLE and app.state.graph_chain is not None
        elif engine == "chain":
            use_graph = False
        else:
            use_graph = bool(app.state.use_langgraph) and LANGGRAPH_AVAILABLE and app.state.graph_chain is not None

        if use_graph:
            answer, sources = answer_with_langgraph(app.state.graph_chain, question)
            logger.info("Q/A (LangGraph)\nQ: %s\nA: %s\nSources: %s", question, answer, list(sources))
        else:
            if app.state.retrieval_chain is None:
                app.state.retrieval_chain = build_retrieval_chain(app.state.llm, app.state.vectorstore, cfg.TOP_K, logger)
            answer, sources = answer_with_sources(app.state.retrieval_chain, question)
            logger.info("Q/A (RetrievalQA)\nQ: %s\nA: %s\nSources: %s", question, answer, list(sources))

        return {"answer": answer, "sources": list(sources)}
    except Exception as e:
        logger.exception("Failed to answer question")
        raise HTTPException(status_code=500, detail=f"Answering failed: {e}")
