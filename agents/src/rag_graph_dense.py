from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from query_processing import classify_query, rewrite_query, process_query
from retrieval import dense_search


class RAGState(TypedDict, total=False):
    question: str
    session_id: Optional[str]
    memory_context: str

    llm: Any
    embedding_model: Any
    embedding_model_name: str
    vectorstores: Any

    use_reranking: bool
    rerank_method: str

    mode: str
    rewritten_query: str
    docs: List[Dict[str, Any]]
    answer: str
    contexts: List[str]


def classify_and_rewrite_node(state: RAGState) -> RAGState:
    q = state["question"]
    llm = state["llm"]
    memory_context = state.get("memory_context", "")

    mode = classify_query(llm, q)
    rewritten = rewrite_query(llm, q, memory_context)

    print(f"[DEBUG] raw: {q}")
    print(f"[DEBUG] rewritten: {rewritten}")

    state["mode"] = mode
    state["rewritten_query"] = rewritten
    return state


def retrieve_dense_node(state: RAGState) -> RAGState:
    query = state["rewritten_query"]
    embedding_model = state["embedding_model"]
    embedding_model_name = state["embedding_model_name"]
    vectorstores = state["vectorstores"]
    llm = state["llm"]
    mode = state["mode"]

    use_reranking = state.get("use_reranking", False)
    rerank_method = state.get("rerank_method", "cross_encoder")

    docs = dense_search(
        query=query,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
        vectorstores=vectorstores,
        classification_mode=mode,
        use_reranking=use_reranking,
        llm=llm,
        rerank_method=rerank_method,
    )

    state["docs"] = docs
    return state


def answer_node(state: RAGState) -> RAGState:
    query = state["rewritten_query"]
    llm = state["llm"]
    mode = state["mode"]
    docs = state.get("docs", [])
    memory_context = state.get("memory_context", "")

    answer, contexts = process_query(
        docs=docs,
        query=query,
        llm=llm,
        classification_mode=mode,
        memory_context=memory_context,
    )

    state["answer"] = answer
    state["contexts"] = contexts
    return state


def build_dense_rag_graph():
    g = StateGraph(RAGState)
    g.add_node("classify_and_rewrite", classify_and_rewrite_node)
    g.add_node("retrieve_dense", retrieve_dense_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("classify_and_rewrite")
    g.add_edge("classify_and_rewrite", "retrieve_dense")
    g.add_edge("retrieve_dense", "answer")
    g.add_edge("answer", END)

    return g.compile()
