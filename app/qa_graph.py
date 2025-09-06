# app/qa_graph.py
from __future__ import annotations
from typing import TypedDict, List, Set

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers strictly from the provided context.\n"
    "If the answer cannot be derived from the context, say you don't know.\n"
    "Be concise and cite only facts present in the context.\n\n"
    "<context>\n{context}\n</context>"
)
USER_PROMPT = "Question: {question}"

class QAState(TypedDict, total=False):
    question: str
    docs: List[Document]
    answer: str
    sources: Set[str]

def _format_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = (d.metadata or {}).get("source", "unknown")
        parts.append(f"[{i}] ({src})\n{d.page_content}")
    return "\n\n".join(parts)

def build_langgraph_chain(llm, vectorstore: FAISS, top_k: int, logger=None):
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )

    def retrieve(state: QAState) -> QAState:
        q = state["question"]
        docs = retriever.get_relevant_documents(q)
        if logger:
            logger.info("Retrieved %d docs for question='%s'", len(docs), q)
        return {"docs": docs}

    def generate(state: QAState) -> QAState:
        docs = state.get("docs", [])
        ctx = _format_context(docs)
        msg = prompt.format(context=ctx, question=state["question"])
        # llm.invoke δέχεται ChatPromptValue ή string (ανάλογα το LLM)
        out = llm.invoke(msg.to_messages() if hasattr(msg, "to_messages") else str(msg))
        text = out.content if hasattr(out, "content") else str(out)
        sources = { (d.metadata or {}).get("source", "unknown") for d in docs }
        return {"answer": text, "sources": sources}

    graph = StateGraph(QAState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

def answer_with_langgraph(chain, question: str) -> tuple[str, Set[str]]:
    # το chain επιστρέφει ολόκληρο το state
    result: QAState = chain.invoke({"question": question})
    return result.get("answer", ""), result.get("sources", set())
