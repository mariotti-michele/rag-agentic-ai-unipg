from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from query_processing import classify_query, rewrite_query, process_query, decompose_question, combine_answers
from retrieval import dense_search, bm25_search, hybrid_search
from prompts import simple_prompt_template


class RAGState(TypedDict, total=False):
    question: str
    session_id: Optional[str]
    memory_context: str
    rewritten_query: str
    mode: str

    is_composite: bool
    sub_questions: List[str]
    sub_answers: Annotated[List[Dict[str, Any]], operator.add]

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


class SingleQuestionState(TypedDict, total=False):
    question: str
    memory_context: str
    mode: str
    rewritten_query: str
    
    search_technique: str
    use_reranking: bool
    rerank_method: str
    docs: List[Dict[str, Any]]
    
    answer: str
    contexts: List[str]
    
    llm: Any
    embedding_model: Any
    embedding_model_name: str
    vectorstores: Any
    corpus: Any
    bm25: Any
    nlp: Any
    reranker: Any


def classify_question_node(state: SingleQuestionState) -> SingleQuestionState:
    q = state["question"]
    llm = state["llm"]
    mode = classify_query(llm, q)
    state["mode"] = mode
    return state


def route_classify(state: SingleQuestionState) -> str:
    if state.get("mode") == "semplice":
        return "simple"
    return "rag"


def simple_answer_node(state: SingleQuestionState) -> SingleQuestionState:
    llm = state["llm"]
    q = state["question"]
    prompt = simple_prompt_template.format(question=q)
    ans = llm.invoke(prompt)
    if hasattr(ans, "content"):
        ans = ans.content
    state["answer"] = ans
    state["contexts"] = []
    return state


def rewrite_question_node(state: SingleQuestionState) -> SingleQuestionState:
    q = state["question"]
    llm = state["llm"]
    memory_context = state.get("memory_context", "")
    rewritten = rewrite_query(llm, q, memory_context)
    state["rewritten_query"] = rewritten
    return state


def route_retrieve(state: SingleQuestionState) -> str:
    return state.get("search_technique", "dense")


def retrieve_dense_node(state: SingleQuestionState) -> SingleQuestionState:
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


def retrieve_sparse_node(state: SingleQuestionState) -> SingleQuestionState:
    state["docs"] = bm25_search(
        state["corpus"],
        state["rewritten_query"],
        state["bm25"],
        state["nlp"],
        classification_mode=state["mode"],
    )
    return state


def retrieve_hybrid_node(state: SingleQuestionState) -> SingleQuestionState:
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


def answer_node(state: SingleQuestionState) -> SingleQuestionState:
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


def _build_single_question_subgraph_internal():
    g = StateGraph(SingleQuestionState)
    
    g.add_node("classify", classify_question_node)
    g.add_node("simple_answer", simple_answer_node)
    g.add_node("rewrite", rewrite_question_node)
    g.add_node("retrieve_dense", retrieve_dense_node)
    g.add_node("retrieve_sparse", retrieve_sparse_node)
    g.add_node("retrieve_hybrid", retrieve_hybrid_node)
    g.add_node("answer", answer_node)
    
    g.set_entry_point("classify")
    
    g.add_conditional_edges(
        "classify",
        route_classify,
        {
            "simple": "simple_answer",
            "rag": "rewrite",
        }
    )
    
    g.add_conditional_edges(
        "rewrite",
        route_retrieve,
        {
            "dense": "retrieve_dense",
            "sparse": "retrieve_sparse",
            "hybrid": "retrieve_hybrid",
        }
    )
    
    g.add_edge("simple_answer", END)
    g.add_edge("retrieve_dense", "answer")
    g.add_edge("retrieve_sparse", "answer")
    g.add_edge("retrieve_hybrid", "answer")
    g.add_edge("answer", END)
    
    return g.compile()


SINGLE_QUESTION_SUBGRAPH = _build_single_question_subgraph_internal()


def decompose_node(state: RAGState) -> RAGState:
    q = state["question"]
    llm = state["llm"]
    is_composite, sub_questions = decompose_question(llm, q)
    state["is_composite"] = is_composite
    if is_composite:
        state["sub_questions"] = sub_questions
        state["sub_answers"] = []
    return state


def route_after_decompose(state: RAGState) -> str | list[Send]:
    if state.get("is_composite", False):
        sub_questions = state.get("sub_questions", [])
        return [
            Send(
                "process_subquestion",
                {
                    "idx": idx,
                    "sub_question": sub_q,
                    "memory_context": state.get("memory_context", ""),
                    "search_technique": state.get("search_technique", "dense"),
                    "use_reranking": state.get("use_reranking", False),
                    "rerank_method": state.get("rerank_method", "cross_encoder"),
                    "llm": state["llm"],
                    "embedding_model": state["embedding_model"],
                    "embedding_model_name": state["embedding_model_name"],
                    "vectorstores": state["vectorstores"],
                    "corpus": state.get("corpus"),
                    "bm25": state.get("bm25"),
                    "nlp": state.get("nlp"),
                    "reranker": state.get("reranker"),
                }
            )
            for idx, sub_q in enumerate(sub_questions)
        ]
    return "single"


def process_single_question_wrapper(state: RAGState) -> RAGState:
    subgraph_state: SingleQuestionState = {
        "question": state["question"],
        "memory_context": state.get("memory_context", ""),
        "search_technique": state.get("search_technique", "dense"),
        "use_reranking": state.get("use_reranking", False),
        "rerank_method": state.get("rerank_method", "cross_encoder"),
        "llm": state["llm"],
        "embedding_model": state["embedding_model"],
        "embedding_model_name": state["embedding_model_name"],
        "vectorstores": state["vectorstores"],
        "corpus": state.get("corpus"),
        "bm25": state.get("bm25"),
        "nlp": state.get("nlp"),
        "reranker": state.get("reranker"),
    }
    
    result = SINGLE_QUESTION_SUBGRAPH.invoke(subgraph_state)
    
    state["answer"] = result["answer"]
    state["contexts"] = result["contexts"]
    return state


def process_subquestion_wrapper(state: Dict[str, Any]) -> dict:
    idx = state["idx"]
    sub_q = state["sub_question"]
    print(f"[INFO] Processing sottodomanda n. {idx} in parallelo: {sub_q}")
    
    subgraph_state: SingleQuestionState = {
        "question": sub_q,
        "memory_context": state.get("memory_context", ""),
        "search_technique": state.get("search_technique", "dense"),
        "use_reranking": state.get("use_reranking", False),
        "rerank_method": state.get("rerank_method", "cross_encoder"),
        "llm": state["llm"],
        "embedding_model": state["embedding_model"],
        "embedding_model_name": state["embedding_model_name"],
        "vectorstores": state["vectorstores"],
        "corpus": state.get("corpus"),
        "bm25": state.get("bm25"),
        "nlp": state.get("nlp"),
        "reranker": state.get("reranker"),
    }
    
    result = SINGLE_QUESTION_SUBGRAPH.invoke(subgraph_state)
    
    return {
        "sub_answers": [{
            "idx": idx,
            "question": sub_q,
            "answer": result["answer"],
            "contexts": result["contexts"]
        }]
    }


def combine_answers_node(state: RAGState) -> RAGState:
    llm = state["llm"]
    original_question = state["question"]
    sub_answers = state.get("sub_answers", [])
    
    sub_answers_sorted = sorted(sub_answers, key=lambda x: x.get("idx", 10**9))
    print(f"[INFO] Elenco risposte parziali (testo completo): {[x.get('answer', '') for x in sub_answers_sorted]}")
    
    print(f"[INFO] Combinazione di {len(sub_answers_sorted)} risposte parziali (ordine: {[x.get('idx', '?') for x in sub_answers_sorted]})")
    combined_answer = combine_answers(llm, original_question, sub_answers_sorted)
    
    all_contexts = []
    for item in sub_answers_sorted:
        all_contexts.extend(item.get("contexts", []))
    
    state["answer"] = combined_answer
    state["contexts"] = all_contexts
    return state


def build_rag_graph():
    g = StateGraph(RAGState)

    g.add_node("decompose", decompose_node)
    g.add_node("process_single_question", process_single_question_wrapper)
    g.add_node("process_subquestion", process_subquestion_wrapper)
    g.add_node("combine_answers", combine_answers_node)

    g.set_entry_point("decompose")

    g.add_conditional_edges(
        "decompose",
        route_after_decompose,
        {
            "single": "process_single_question",
            # se route_after_decompose restituisce lista di Send, langgraph esegue process_subquestion in parallelo per ogni sottodomanda
        }
    )

    g.add_edge("process_single_question", END)

    g.add_edge("process_subquestion", "combine_answers")
    g.add_edge("combine_answers", END)

    return g.compile()
