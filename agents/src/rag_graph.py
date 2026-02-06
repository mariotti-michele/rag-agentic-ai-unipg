from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from query_processing import classify_query, rewrite_query, process_query
from retrieval import dense_search, bm25_search, hybrid_search, fallback_retrieve_with_expansion
from prompts import simple_prompt_template, EVAL_ANSWER_PROMPT

class RAGState(TypedDict, total=False):
    question: str
    session_id: Optional[str]
    memory_context: str
    rewritten_query: str
    mode: str

    search_technique: str
    use_reranking: bool
    rerank_method: str
    docs: List[Dict[str, Any]]
    contexts: List[str]
    
    answer: str

    llm: Any
    embedding_model: Any
    embedding_model_name: str
    vectorstores: Any
    corpus: Any
    bm25: Any
    nlp: Any
    reranker: Any

    # fallback/eval
    needs_fallback: bool
    force_fallback: bool
    fallback_used: bool
    fallback_reason: str
    ui_message: str



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


def evaluate_node(state: RAGState) -> RAGState:
    q = state.get("question", "")
    ans = state.get("answer", "") or ""
    n_sources = len(state.get("contexts", []) or [])

    # Heuristics veloci (prima dell’LLM)
    low_quality = (
        n_sources == 0 or
        "non presente nei documenti" in ans.lower() or
        "non sono in grado di rispondere" in ans.lower() or
        "parziale e poco affidabile" in ans.lower()
    )

    if low_quality:
        state["needs_fallback"] = True
        state["fallback_reason"] = "heuristic_low_quality"
        state["ui_message"] = "Sto cercando più a fondo..."

        print(
            f"[EVAL] low_quality=True | sid={state.get('session_id')} | "
            f"sources={n_sources} | reason={state['fallback_reason']}"
        )
        return state

    print(
        f"[EVAL] low_quality=False | sid={state.get('session_id')} | "
        f"sources={n_sources}"
    )


    # Valutazione LLM (agente “critic”)
    llm = state.get("llm")
    if llm:
        out = llm.invoke(EVAL_ANSWER_PROMPT.format(question=q, answer=ans, n_sources=n_sources))
        if hasattr(out, "content"):
            out = out.content
        verdict = str(out).strip()

        if verdict.startswith("FALLBACK"):
            state["needs_fallback"] = True
            state["fallback_reason"] = verdict[:200]
            state["ui_message"] = "Sto cercando più a fondo..."
        else:
            state["needs_fallback"] = False
            state["fallback_reason"] = ""
            state["ui_message"] = ""
    else:
        state["needs_fallback"] = False
        state["fallback_reason"] = ""
        state["ui_message"] = ""

    if not state.get("needs_fallback", False):
        state["fallback_used"] = False

    return state


def fallback_retrieve_node(state: RAGState) -> RAGState:
    # segna fallback
    state["fallback_used"] = True
    print(f"[FALLBACK] entered | sid={state.get('session_id')} | technique={state.get('search_technique')}")

    docs = fallback_retrieve_with_expansion(
        query=state["rewritten_query"],
        search_technique=state.get("search_technique", "dense"),
        embedding_model=state["embedding_model"],
        embedding_model_name=state["embedding_model_name"],
        vectorstores=state["vectorstores"],
        corpus=state["corpus"],
        bm25=state["bm25"],
        nlp=state["nlp"],
        llm=state.get("llm"),
        reranker=state.get("reranker"),
        final_top_k=5,
        candidate_k=25,  # <<< QUI controlli “più di 10 chunk”
    )
    state["docs"] = docs
    return state


def fallback_answer_node(state: RAGState) -> RAGState:
    # riusa la stessa process_query che già usi :contentReference[oaicite:8]{index=8}
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


def route_after_evaluate(state: RAGState) -> str:
    if not state.get("needs_fallback", False):
        return "end"

    # fallback necessario, ma NON forzato → fermati qui
    if not state.get("force_fallback", False):
        return "end"

    # fallback necessario E forzato → esegui fallback
    return "fallback"



def build_rag_graph():
    g = StateGraph(RAGState)

    g.add_node("classify", classify_and_rewrite_node)
    g.add_node("simple_answer", simple_answer_node)

    g.add_node("retrieve_dense", retrieve_dense_node)
    g.add_node("retrieve_sparse", retrieve_sparse_node)
    g.add_node("retrieve_hybrid", retrieve_hybrid_node)

    g.add_node("answer", answer_node)

    g.add_node("evaluate", evaluate_node)
    g.add_node("fallback_retrieve", fallback_retrieve_node)
    g.add_node("fallback_answer", fallback_answer_node)

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

    # answer -> evaluate -> (end | fallback path)
    g.add_edge("answer", "evaluate")

    g.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "end": END,
            "fallback": "fallback_retrieve",
        },
    )

    g.add_edge("fallback_retrieve", "fallback_answer")
    g.add_edge("fallback_answer", END)

    return g.compile()
