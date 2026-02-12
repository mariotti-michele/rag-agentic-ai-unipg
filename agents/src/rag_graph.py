from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from query_processing import classify_query, rewrite_query, process_query, decompose_question, combine_answers
from fallback import fallback_retrieve_with_expansion
from retrieval import dense_search, bm25_search, hybrid_search
from prompts import simple_prompt_template, EVAL_ANSWER_PROMPT


class RAGState(TypedDict, total=False):
    question: str
    session_id: Optional[str]
    memory_context: str
    mode: str

    is_composite: bool
    sub_questions: List[str]
    sub_answers: Annotated[List[Dict[str, Any]], operator.add]

    search_technique: str
    use_reranking: bool
    rerank_method: str
    docs: List[Dict[str, Any]]
    #contexts: List[str]
    references: List[Dict[str, Any]]
    
    answer: str

    llm: Any
    embedding_model: Any
    embedding_model_name: str
    vectorstores: Any
    corpus: Any
    bm25: Any
    nlp: Any
    reranker: Any

    force_fallback: bool
    emit: Any


class SingleQuestionState(TypedDict, total=False):
    question: str
    session_id: Optional[str]
    memory_context: str
    mode: str
    rewritten_query: str
    
    search_technique: str
    use_reranking: bool
    rerank_method: str
    docs: List[Dict[str, Any]]
    
    answer: str
    #contexts: List[str]
    references: List[Dict[str, Any]]
    
    llm: Any
    embedding_model: Any
    embedding_model_name: str
    vectorstores: Any
    corpus: Any
    bm25: Any
    nlp: Any
    reranker: Any

    needs_fallback: bool
    force_fallback: bool
    fallback_reason: str
    emit: Any


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
    emit = state.get("emit")
    if emit:
        emit({
                "type": "status", 
                "message": f"Sto rispondendo alla tua domanda di carattere generale senza cercare nei documenti"
            })        
    llm = state["llm"]
    q = state["question"]
    prompt = simple_prompt_template.format(question=q)
    ans = llm.invoke(prompt)
    if hasattr(ans, "content"):
        ans = ans.content
    state["answer"] = ans
    #state["contexts"] = []
    state["references"] = []
    print(f"[INFO] Risposta generata per domanda semplice: {ans}")
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
    emit = state.get("emit")
    if emit:
        emit({
                "type": "status", 
                "message": f"Sto cercando nei documenti la risposta alla tua domanda"
            })
            

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
    emit = state.get("emit")
    argument = state.get("mode", "generale")
    if state.get("mode") == "rag":
        argument = "generale"
    if emit:
        emit({
                "type": "status", 
                "message": f"Sto generando la risposta alla tua domanda sull'argomento: {argument}"
            })
    
    #answer, contexts = process_query(
    answer, references = process_query(
        docs=state.get("docs", []),
        query=state["rewritten_query"],
        llm=state["llm"],
        classification_mode=state["mode"],
        memory_context=state.get("memory_context", ""),
    )
    state["answer"] = answer
    #state["contexts"] = contexts
    state["references"] = references
    return state




def evaluate_node(state: SingleQuestionState) -> SingleQuestionState:
    q = state.get("question", "")
    ans = state.get("answer", "") or ""
    #n_sources = len(state.get("contexts", []) or [])
    n_sources = len(state.get("references", []) or [])

    state["needs_fallback"] = False
    state["fallback_reason"] = ""

    low_quality = (
        (n_sources == 0 and state.get("mode") != "semplice") or
        "non presente nei documenti" in ans.lower() or
        "non sono in grado di rispondere" in ans.lower() or
        "parziale e poco affidabile" in ans.lower()
    )

    if low_quality:
        state["needs_fallback"] = True
        state["fallback_reason"] = "heuristic_low_quality"
        state["mode"] = "rag"
        emit = state.get("emit")
        if emit:
            emit({"type": "status", "message": "Sto cercando più a fondo..."})
        print(
            f"[EVAL] low_quality=True | sid={state.get('session_id')} | "
            f"sources={n_sources} | reason={state['fallback_reason']}"
        )
        return state

    llm = state.get("llm")
    if llm:
        out = llm.invoke(EVAL_ANSWER_PROMPT.format(question=q, answer=ans, n_sources=n_sources))
        if hasattr(out, "content"):
            out = out.content
        verdict = str(out).strip()

        if verdict.startswith("FALLBACK"):
            state["needs_fallback"] = True
            state["fallback_reason"] = verdict
            state["mode"] = "rag"
            emit = state.get("emit")
            if emit:
                emit({"type": "status", "message": "Sto cercando più a fondo..."})
        else:
            state["needs_fallback"] = False
            state["fallback_reason"] = ""
    else:
        state["needs_fallback"] = False
        state["fallback_reason"] = ""

    return state


def route_after_evaluate(state: SingleQuestionState) -> str:
    if not state.get("needs_fallback", False) and not state.get("force_fallback", False):
        return "end"
    return "fallback"


def fallback_retrieve_node(state: SingleQuestionState) -> SingleQuestionState:
    print(f"[FALLBACK] entered | sid={state.get('session_id')} | technique={state.get('search_technique')} | reason={state.get('fallback_reason')} | answer_to_improve={state.get('answer')}")

    docs = fallback_retrieve_with_expansion(
        query=state["rewritten_query"],
        embedding_model=state["embedding_model"],
        embedding_model_name=state["embedding_model_name"],
        vectorstores=state["vectorstores"],
        llm=state.get("llm"),
        reranker=state.get("reranker"),
    )
    state["docs"] = docs
    return state


def fallback_answer_node(state: SingleQuestionState) -> SingleQuestionState:
    #answer, contexts = process_query(
    answer, references = process_query(
        docs=state.get("docs", []),
        query=state["rewritten_query"],
        llm=state["llm"],
        classification_mode=state["mode"],
        memory_context=state.get("memory_context", ""),
    )
    state["answer"] = answer
    #state["contexts"] = contexts
    state["references"] = references
    print(f"[FALLBACK] nuova risposta generata: {answer}\nContesti usati: {len(contexts)}")
    return state

def secondary_evaluate_node(state: SingleQuestionState) -> SingleQuestionState:
    ans = state.get("answer", "") or ""
    n_sources = len(state.get("contexts", []) or [])

    state["needs_fallback"] = False
    state["fallback_reason"] = ""

    low_quality = (
        (n_sources == 0 and state.get("mode") != "semplice") or
        "non presente nei documenti" in ans.lower() or
        "non sono in grado di rispondere" in ans.lower()
    )

    if low_quality:
        state["needs_fallback"] = True
        state["fallback_reason"] = "heuristic_low_quality on second eval"
        state["mode"] = "rag"
        emit = state.get("emit")
        if emit:
            emit({"type": "status", "message": "Non riesco a trovare la risposta, faccio un ultimo tentativo..."})
        print(
            f"[Secondary EVAL] low_quality=True | sid={state.get('session_id')} | "
            f"sources={n_sources} | reason={state['fallback_reason']}"
        )
    return state

def route_after_secondary_evaluate(state: SingleQuestionState) -> str:
    if not state.get("needs_fallback", False):
        return "end"
    return "fallback"


def retrieval_without_memory_context_node(state: SingleQuestionState) -> SingleQuestionState:
    print(f"[INFO] Riprovo facendo retrieval senza contesto di memoria | sid={state.get('session_id')}")
    state["rewritten_query"] = state["question"]
    return retrieve_dense_node(state)


def answer_after_secondary_evaluate_node(state: SingleQuestionState) -> SingleQuestionState:
    answer, contexts = process_query(
        docs=state.get("docs", []),
        query=state["question"],
        llm=state["llm"],
        classification_mode=state["mode"],
        memory_context=state.get("memory_context", ""),
    )
    state["answer"] = answer
    state["contexts"] = contexts
    print(f"[INFO] Risposta generata dopo secondary evaluate: {answer}\nContesti usati: {len(contexts)}")
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

    g.add_node("evaluate", evaluate_node)
    g.add_node("fallback_retrieve", fallback_retrieve_node)
    g.add_node("fallback_answer", fallback_answer_node)

    g.add_node("secondary_evaluate", secondary_evaluate_node)
    g.add_node("retrieval_without_memory_context", retrieval_without_memory_context_node)
    g.add_node("answer_after_secondary_evaluate", answer_after_secondary_evaluate_node)
    
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
    g.add_edge("fallback_answer", "secondary_evaluate")
    g.add_conditional_edges(
        "secondary_evaluate",
        route_after_secondary_evaluate,
        {
            "end": END,
            "fallback": "retrieval_without_memory_context",
        },
    )
    g.add_edge("retrieval_without_memory_context", "answer_after_secondary_evaluate")
    g.add_edge("answer_after_secondary_evaluate", END)
    
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
                    "session_id": state.get("session_id"),
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
                    "force_fallback": state.get("force_fallback", False),
                    "emit": state.get("emit"),
                }
            )
            for idx, sub_q in enumerate(sub_questions)
        ]
    return "single"


def process_single_question_wrapper(state: RAGState) -> RAGState:
    subgraph_state: SingleQuestionState = {
        "question": state["question"],
        "session_id": state.get("session_id"),
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
        "force_fallback": state.get("force_fallback", False),
        "emit": state.get("emit"),
    }
    
    result = SINGLE_QUESTION_SUBGRAPH.invoke(subgraph_state)
    
    state["answer"] = result["answer"]
    #state["contexts"] = result["contexts"]
    state["references"] = result.get("references", [])
    return state


def process_subquestion_wrapper(state: Dict[str, Any]) -> dict:
    idx = state["idx"]
    sub_q = state["sub_question"]
    print(f"[INFO] Processing sottodomanda n. {idx} in parallelo: {sub_q}")
    
    subgraph_state: SingleQuestionState = {
        "question": sub_q,
        "session_id": state.get("session_id"),
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
        "force_fallback": state.get("force_fallback", False),
        "emit": state.get("emit"),
    }
    
    result = SINGLE_QUESTION_SUBGRAPH.invoke(subgraph_state)
    
    return {
        "sub_answers": [{
            "idx": idx,
            "question": sub_q,
            "answer": result["answer"],
            #"contexts": result["contexts"]
            "references": result.get("references", [])
        }]
    }


def combine_answers_node(state: RAGState) -> RAGState:
    llm = state["llm"]
    original_question = state["question"]
    sub_answers = state.get("sub_answers", [])
    
    sub_answers_sorted = sorted(sub_answers, key=lambda x: x.get("idx", 10**9))

    emit = state.get("emit")
    if emit:
        emit({
            "type": "status",
            "message": f"Sto combinando {len(sub_answers_sorted)} risposte parziali..."
        })
    
    print(f"[INFO] Combinazione di {len(sub_answers_sorted)} risposte parziali (ordine: {[x.get('idx', '?') for x in sub_answers_sorted]})")
    combined_answer = combine_answers(llm, original_question, sub_answers_sorted)
    
    #all_contexts = []
    #for item in sub_answers_sorted:
        #all_contexts.extend(item.get("contexts", []))

    all_refs = []
    seen = set()
    for item in sub_answers_sorted:
        for r in item.get("references", []) or []:
            key = (r.get("url"), r.get("title"), r.get("section"))
            if key in seen:
                continue
            seen.add(key)
            all_refs.append(r)
    
    state["answer"] = combined_answer
    #state["contexts"] = all_contexts
    state["references"] = all_refs
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