# utils.py
# Environment/config + logging helpers

from __future__ import annotations
import os
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

def load_env():
    """Load variables from .env if python-dotenv is installed; otherwise no-op."""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass


class Config:
    # Directories
    DATA_DIR: str = os.environ.get("DATA_DIR", "data")
    INDEX_DIR: str = os.environ.get("INDEX_DIR", "data/index")
    LOG_DIR: str = os.environ.get("LOG_DIR", "logs")

    # RAG params
    CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "150"))
    TOP_K: int = int(os.environ.get("TOP_K", "4"))
    TEMPERATURE: float = float(os.environ.get("TEMPERATURE", "0.0"))

    # OpenAI models
    OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBED_MODEL: str = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    # HuggingFace fallbacks
    HF_EMBED_MODEL: str = os.environ.get("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    HF_LLM_MODEL: str = os.environ.get("HF_LLM_MODEL", "google/flan-t5-base")


def ensure_dirs(paths):
    if isinstance(paths, (list, tuple)):
        for p in paths:
            Path(p).mkdir(parents=True, exist_ok=True)
    else:
        Path(paths).mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "app", log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger
