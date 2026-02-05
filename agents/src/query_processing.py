from prompts import EXAM_CALENDAR_PROMPT, GRADUATION_CALENDAR_PROMPT, MODULES_PROMPT, RAG_PROMPT, TIMETABLE_PROMPT, CLASSIFIER_PROMPT, TIMETABLE_PROMPT, simple_prompt_template, QUERY_REWRITE_PROMPT
from retrieval import bm25_search, dense_search, hybrid_search


def build_context(docs: list) -> str:
    context = ""
    for i, d in enumerate(docs, 1):
        section = f" | Sezione: {d.get('section_path','')}" if d.get("section_path") else ""
        context += f"[Fonte {i}] ({d.get('collection','N/A')}){section}\n{d['text']}\n\n"
    return context

def should_rewrite(question: str) -> bool:
    q = question.strip().lower()

    followup_markers = [
        "questo", "questa", "quello", "quella", "questi", "queste", "quelli", "quelle",
        "lui", "lei", "esso", "essa",
        "e invece", "e quello", "e questa", "ok e", "allora e", "per quello", "riguardo a"
    ]

    domain_markers = [
        "cfu", "crediti",
        "date", "appelli", "esame", "esami", "quando",
        "orario", "lezione", "aula",
        "prof", "docente",
        "calendario lauree", "laurea", "sessione"
    ]

    is_followup = any(m in q for m in followup_markers) or len(q) <= 45

    in_domain = any(m in q for m in domain_markers)

    return is_followup and in_domain


def rewrite_query(llm, question: str, memory_context: str) -> str:
    if not memory_context or not memory_context.strip():
        return question

    if not should_rewrite(question):
        return question

    prompt = QUERY_REWRITE_PROMPT.format(
        memory=memory_context,
        question=question
    )

    out = llm.invoke(prompt)
    if hasattr(out, "content"):
        out = out.content

    rewritten = str(out).strip().strip('"').strip("'")

    if not rewritten or len(rewritten) < 3:
        return question

    if rewritten.lower() == question.strip().lower():
        return question

    return rewritten


def get_llm_answer(context: str, query: str, llm, prompt_template, memory_context: str = "") -> str:
    prompt = prompt_template.format(context=context, question=query)

    if memory_context:
        prompt = memory_context.strip() + "\n\n" + prompt

    answer = llm.invoke(prompt)
    if hasattr(answer, "content"):
        answer = answer.content
    return answer


def process_query(docs: list, query: str, llm, classification_mode, memory_context: str = "") -> tuple[str, list]:
    if not docs:
        return "Non presente nei documenti", []
    context = build_context(docs)
    prompt_template = RAG_PROMPT
    if classification_mode == "orario":
        prompt_template = TIMETABLE_PROMPT
    elif classification_mode == "calendario esami":
        prompt_template = EXAM_CALENDAR_PROMPT
    elif classification_mode == "insegnamenti":
        prompt_template = MODULES_PROMPT
    elif classification_mode == "calendario lauree":
        prompt_template = GRADUATION_CALENDAR_PROMPT
    answer = get_llm_answer(context, query, llm, prompt_template, memory_context)
    return answer, [d["text"] for d in docs]


def answer_query_dense(query: str, embedding_model, embedding_model_name: str, vectorstores, llm, classification_mode, use_reranking=False, rerank_method="cross_encoder", memory_context: str = ""):
    dense_docs = dense_search(query, embedding_model, embedding_model_name, vectorstores, classification_mode=classification_mode, use_reranking=use_reranking, llm=llm, rerank_method=rerank_method)
    return process_query(dense_docs, query, llm, classification_mode, memory_context)

def answer_query_bm25(query: str, corpus, bm25, nlp, llm, classification_mode, memory_context: str = ""):
    sparse_docs = bm25_search(corpus, query, bm25, nlp, classification_mode=classification_mode)
    return process_query(sparse_docs, query, llm, classification_mode, memory_context)

def answer_query_hybrid(query: str, embedding_model, embedding_model_name, vectorstores, corpus, bm25, nlp, llm, classification_mode, use_reranking=False, rerank_method="cross_encoder", memory_context: str = ""):
    merged_docs = hybrid_search(query, embedding_model, embedding_model_name, vectorstores, corpus, bm25, nlp, classification_mode=classification_mode, use_reranking=use_reranking, llm=llm, rerank_method=rerank_method)
    return process_query(merged_docs, query, llm, classification_mode, memory_context)


def classify_query(llm, query: str) -> str:
    try:
        classification = llm.invoke(CLASSIFIER_PROMPT.format(question=query))
        if hasattr(classification, "content"):
            classification = classification.content
        classification = str(classification).strip().lower()

        if "semplice" in classification:
            print("[INFO] Query classificata come semplice.")
            return "semplice"
        elif "orario" in classification:
            print("[INFO] Query classificata come orario.")
            return "orario"
        elif "calendario esami" in classification:
            print("[INFO] Query classificata come calendario esami.")
            return "calendario esami"
        elif "insegnamenti" in classification:
            print("[INFO] Query classificata come insegnamenti.")
            return "insegnamenti"
        elif "calendario lauree" in classification:
            print("[INFO] Query classificata come calendario lauree.")
            return "calendario lauree"
        else:
            return "rag"
    except Exception as e:
        print(f"[WARN] Errore classificazione query: {e}")
        return "rag"


def generate_answer(llm, query: str, search_technique, embedding_model, embedding_model_name, vectorstores, corpus, bm25, nlp, use_reranking=False, rerank_method="cross_encoder", memory_context: str = ""):
    mode = classify_query(llm, query)

    rewritten_query = rewrite_query(llm, query, memory_context)
    print(f"[DEBUG] raw: {query}")
    print(f"[DEBUG] rewritten: {rewritten_query}")

    if mode == "semplice":
        prompt = simple_prompt_template.format(question=query)
        answer = llm.invoke(prompt)
        if hasattr(answer, "content"):
            answer = answer.content
        return answer, [], mode
    else:
        answer, contexts = None, []
        if search_technique == "dense":
            answer, contexts = answer_query_dense(rewritten_query, embedding_model, embedding_model_name, vectorstores, llm, mode, use_reranking, rerank_method, memory_context)
        elif search_technique == "sparse":
            answer, contexts = answer_query_bm25(rewritten_query, corpus, bm25, nlp, llm, mode, memory_context)
        elif search_technique == "hybrid":
            answer, contexts = answer_query_hybrid(rewritten_query, embedding_model, embedding_model_name, vectorstores, corpus, bm25, nlp, llm, mode, use_reranking, rerank_method, memory_context)
    return answer, contexts, mode


_rag_graph = None

def _get_rag_graph():
    global _rag_graph
    if _rag_graph is None:
        from rag_graph import build_rag_graph
        _rag_graph = build_rag_graph()
    return _rag_graph

def generate_answer_via_graph(state: dict):
    graph = _get_rag_graph()
    result = graph.invoke(state)
    return result["answer"], result.get("contexts", []), result.get("mode", "rag")
