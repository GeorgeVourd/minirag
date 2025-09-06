# ingestion.py
# Utilities for loading, splitting, and persisting documents & FAISS vectorstore.

from __future__ import annotations
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

ALLOWED_EXTS = {".txt", ".md"}


def load_file_to_documents(*, text: str, source: str) -> List[Document]:
    """Wrap raw text into a single Document with 'source' metadata."""
    return [Document(page_content=text, metadata={"source": source})]


def split_text_documents(
    docs: List[Document],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(docs)
    # Ensure 'source' metadata is present on all chunks
    for c in chunks:
        c.metadata["source"] = c.metadata.get("source") or "unknown"
    return chunks


def create_vectorstore_from_docs(docs: List[Document], embeddings, logger=None) -> FAISS:
    if logger:
        logger.info("Creating FAISS index from %d chunks...", len(docs))
    vs = FAISS.from_documents(docs, embeddings)
    return vs


def save_vectorstore(vs: FAISS, index_dir: str, logger=None) -> None:
    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)
    # Saves index + docstore in 'index.faiss' and 'index.pkl' under index_dir
    vs.save_local(index_dir)
    if logger:
        logger.info("Saved FAISS index to %s", path.resolve())


def load_vectorstore(index_dir: str, embeddings, logger=None) -> FAISS | None:
    try:
        path = Path(index_dir)
        if not path.exists() or not any(path.iterdir()):
            return None
        # allow_dangerous_deserialization=True is required for loading pickled docstore in many environments.
        vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        if logger:
            logger.info("Loaded FAISS index from %s", path.resolve())
        return vs
    except Exception as e:
        if logger:
            logger.warning("Could not load FAISS index from %s: %s", index_dir, e)
        return None
