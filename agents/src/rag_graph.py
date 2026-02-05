from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from query_processing import classify_query, rewrite_query, process_query
from retrieval import dense_search, bm25_search, hybrid_search
from prompts import simple_prompt_template

class RAGState(TypedDict, total=False):
    question: str
    session_id: Optional[str]
    memory_context: str
    search_technique: str

    llm: Any
    embedding_model: Any
    embedding_model_name: str
    vectorstores: Any
    corpus: Any
    bm25: Any
    nlp: Any

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

    state["mode"] = mode
    state["rewritten_query"] = rewritten
    return state

def route_after_classify(state: RAGState) -> str:
    if state.get("mode") == "semplice":
        return "simple"
    return state.get("search_technique", "dense")

def simple_answer_node(state: RAGState) -> RAGState:
    llm = state["llm"]
    q = state["question"]
    prompt = simple_prompt_template.format(question=q)
    ans = llm.invoke(prompt)
    if hasattr(ans, "content"):
        ans = ans.content
    state["answer"] = ans
    state["contexts"] = []
    return state

def retrieve_dense_node(state: RAGState) -> RAGState:
    state["docs"] = dense_search(
        query=state["rewritten_query"],
        embedding_model=state["embedding_model"],
        embedding_model_name=state["embedding_model_name"],
        vectorstores=state["vectorstores"],
        classification_mode=state["mode"],
        use_reranking=state.get("use_reranking", False),
        llm=state["llm"],
        rerank_method=state.get("rerank_method", "cross_encoder"),
        reranker=state.get("reranker", None),
    )
    return state

def retrieve_sparse_node(state: RAGState) -> RAGState:
    state["docs"] = bm25_search(
        state["corpus"],
        state["rewritten_query"],
        state["bm25"],
        state["nlp"],
        classification_mode=state["mode"],
    )
    return state

def retrieve_hybrid_node(state: RAGState) -> RAGState:
    state["docs"] = hybrid_search(
        query=state["rewritten_query"],
        embedding_model=state["embedding_model"],
        embedding_model_name=state["embedding_model_name"],
        vectorstores=state["vectorstores"],
        corpus=state["corpus"],
        bm25=state["bm25"],
        nlp=state["nlp"],
        classification_mode=state["mode"],
        use_reranking=state.get("use_reranking", False),
        llm=state["llm"],
        rerank_method=state.get("rerank_method", "cross_encoder"),
        reranker=state.get("reranker", None),
    )
    return state

def answer_node(state: RAGState) -> RAGState:
    answer, contexts = process_query(
        docs=state.get("docs", []),
        query=state["rewritten_query"],
        llm=state["llm"],
        classification_mode=state["mode"],
        memory_context=state.get("memory_context", ""),
    )
    state["answer"] = answer
    state["contexts"] = contexts
    return state

def build_rag_graph():
    g = StateGraph(RAGState)

    g.add_node("classify", classify_and_rewrite_node)
    g.add_node("simple_answer", simple_answer_node)

    g.add_node("retrieve_dense", retrieve_dense_node)
    g.add_node("retrieve_sparse", retrieve_sparse_node)
    g.add_node("retrieve_hybrid", retrieve_hybrid_node)

    g.add_node("answer", answer_node)

    g.set_entry_point("classify")

    g.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "simple": "simple_answer",
            "dense": "retrieve_dense",
            "sparse": "retrieve_sparse",
            "hybrid": "retrieve_hybrid",
        },
    )

    g.add_edge("simple_answer", END)
    g.add_edge("retrieve_dense", "answer")
    g.add_edge("retrieve_sparse", "answer")
    g.add_edge("retrieve_hybrid", "answer")
    g.add_edge("answer", END)

    return g.compile()
