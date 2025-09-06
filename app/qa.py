# qa.py
# Build embeddings, LLM, and retrieval chain; run Q/A with sources.

from __future__ import annotations
import os
from typing import Iterable, Set, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Embeddings & LLMs
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline as hf_pipeline


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers the user's question strictly using the provided context.\n"
    "If the answer cannot be derived from the context, say you don't know.\n"
    "Cite only the facts present in the context.\n"
    "Answer concisely.\n\n"
    "<context>\n{context}\n</context>"
)

USER_PROMPT = "Question: {input}"


def get_embeddings(cfg, logger=None):
    """Return an embeddings object. Prefers OpenAI; falls back to HuggingFace if no key."""
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        model = cfg.OPENAI_EMBED_MODEL
        if logger:
            logger.info(f"Using OpenAI embeddings: {model}")
        return OpenAIEmbeddings(model=model)
    # Fallback to HuggingFace
    model = cfg.HF_EMBED_MODEL
    if logger:
        logger.info(f"Using HuggingFace embeddings: {model}")
    return HuggingFaceEmbeddings(model_name=model)


def get_llm(cfg, logger=None):
    """Return an LLM object. Prefers OpenAI Chat model; falls back to HuggingFace local model."""
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        model = cfg.OPENAI_MODEL
        if logger:
            logger.info(f"Using OpenAI chat model: {model}")
        return ChatOpenAI(model=model, temperature=cfg.TEMPERATURE)
    # Fallback to HF local (seq2seq) model suitable for instruction following
    model_name = cfg.HF_LLM_MODEL
    if logger:
        logger.info(f"Using HuggingFace local model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = hf_pipeline(
        "text2text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=512,
    )
    return HuggingFacePipeline(pipeline=gen)


def build_retrieval_chain(llm, vectorstore: FAISS, top_k: int, logger=None):
    """Create a retrieval chain that returns both 'answer' and 'context' docs."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    if logger:
        logger.info(f"Retriever configured with top_k={top_k}")
    chain = create_retrieval_chain(retriever, doc_chain)
    return chain


def answer_with_sources(chain, question: str) -> tuple[str, Set[str]]:
    """Run the retrieval chain and return (answer, unique_sources)."""
    res = chain.invoke({"input": question})
    answer: str = res.get("answer", "") or res.get("result", "")
    context_docs: Iterable[Document] = res.get("context", []) or res.get("source_documents", [])
    sources: Set[str] = set()
    for d in context_docs:
        src = (d.metadata or {}).get("source") or "unknown"
        sources.add(src)
    return answer, sources
