from prompts import EXAM_CALENDAR_PROMPT, GRADUATION_CALENDAR_PROMPT, MODULES_PROMPT, RAG_PROMPT, TIMETABLE_PROMPT, CLASSIFIER_PROMPT, QUERY_REWRITE_PROMPT, QUESTION_DECOMPOSITION_PROMPT, ANSWER_COMBINATION_PROMPT, QUERY_EXPANSION_PROMPT, MULTI_QUERY_PROMPT

def build_context(docs: list) -> str:
    context = ""
    for i, d in enumerate(docs, 1):
        section = f" | Sezione: {d.get('section_path','')}" if d.get("section_path") else ""
        context += f"[Fonte {i}] ({d.get('collection','N/A')}){section}\n{d['text']}\n\n"
    return context


def rewrite_query(llm, question: str, memory_context: str) -> str:
    if not memory_context or not memory_context.strip():
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

    print(f"[INFO] Query riscritta: {rewritten}\n")
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


def decompose_question(llm, question: str) -> tuple[bool, list[str]]:
    try:
        prompt = QUESTION_DECOMPOSITION_PROMPT.format(question=question)
        response = llm.invoke(prompt)
        if hasattr(response, "content"):
            response = response.content
        
        response_text = str(response).strip()
        
        if "SINGOLA" in response_text.upper():
            print("[INFO] Domanda identificata come SINGOLA")
            return False, []
        
        if "MULTIPLA" in response_text.upper():
            print("[INFO] Domanda identificata come MULTIPLA")
            lines = response_text.split("\n")
            sub_questions = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("- ")):
                    question_text = line
                    for prefix in [".", ")", "-"]:
                        if prefix in question_text[:5]:
                            parts = question_text.split(prefix, 1)
                            if len(parts) > 1:
                                question_text = parts[1].strip()
                                break
                    if question_text and len(question_text) > 3:
                        sub_questions.append(question_text)
            
            if sub_questions:
                print(f"[INFO] Identificate {len(sub_questions)} sottodomande")
                print(f"[DEBUG] Sottodomande: {sub_questions}")
                return True, sub_questions
        
        print("[WARN] Impossibile determinare se la domanda è composta, trattata come singola")
        return False, []
    
    except Exception as e:
        print(f"[WARN] Errore nella decomposizione della domanda: {e}")
        return False, []


def combine_answers(llm, original_question: str, sub_answers: list) -> str:
    try:
        partial_answers_text = ""
        for i, item in enumerate(sub_answers, 1):
            partial_answers_text += f"\nDomanda {i}: {item['question']}\n"
            partial_answers_text += f"Risposta {i}: {item['answer']}\n"
        
        prompt = ANSWER_COMBINATION_PROMPT.format(
            original_question=original_question,
            partial_answers=partial_answers_text
        )
        
        response = llm.invoke(prompt)
        if hasattr(response, "content"):
            response = response.content
        
        combined_answer = str(response).strip()
        print("[INFO] Risposte combinate con successo")
        return combined_answer
    
    except Exception as e:
        print(f"[WARN] Errore nella combinazione delle risposte: {e}")
        fallback = ""
        for i, item in enumerate(sub_answers, 1):
            fallback += f"{item['answer']}\n\n"
        return fallback.strip()


def semantic_query_expansion(llm, query: str) -> str:
    """Una singola query espansa semanticamente con LLM."""
    if not llm:
        return query
    out = llm.invoke(QUERY_EXPANSION_PROMPT.format(query=query))
    if hasattr(out, "content"):
        out = out.content
    q2 = str(out).strip().strip('"')
    return q2 if q2 else query


def multi_query_expansion(llm, query: str, n: int = 3) -> list[str]:
    """Ritorna una lista di query alternative (multi-query)."""
    if not llm:
        return [query]
    out = llm.invoke(MULTI_QUERY_PROMPT.format(query=query))
    if hasattr(out, "content"):
        out = out.content
    lines = [l.strip().strip('"') for l in str(out).splitlines() if l.strip()]
    # assicura fallback sensato
    qs = []
    for l in lines:
        if l and l.lower() != query.lower():
            qs.append(l)
    if not qs:
        qs = [query]
    return qs[:n]


def deep_rerank_with_cross_encoder(query: str, hits: list[dict], reranker=None, top_k: int = 5, rerank_k: int = 25) -> list[dict]:
    """
    NUOVA: rerank "profondo" solo per fallback.
    - hits: lista dict con campo "text"
    - rerank_k: quanti candidati massimo passare al reranker
    """
    if not hits:
        return []
    if not reranker:
        # se non ho reranker, fallback "soft": taglio e basta
        return hits[:top_k]

    candidates = hits[:min(len(hits), rerank_k)]
    docs_text = [h.get("text", "") for h in candidates]

    try:
        scores = reranker.rerank(query=query, docs=docs_text)  # ritorna lista float :contentReference[oaicite:4]{index=4}
        for h, s in zip(candidates, scores):
            h["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return candidates[:top_k]
    except Exception as e:
        print(f"[WARN] deep_rerank fallito: {e}")
        return hits[:top_k]


def fallback_retrieve_with_expansion(
    query: str,
    search_technique: str,
    embedding_model,
    embedding_model_name: str,
    vectorstores,
    corpus,
    bm25,
    nlp,
    llm=None,
    reranker=None,
    final_top_k: int = 5,
    candidate_k: int = 25,
) -> list[dict]:
    """
    Recupero da usare SOLO nel fallback:
    - crea query espansa + multi-query
    - fa retrieval più largo (candidate_k)
    - fa deep rerank e ritorna final_top_k
    """
    # 1) query expansion
    q_sem = semantic_query_expansion(llm, query)
    q_multi = multi_query_expansion(llm, query, n=3)
    queries = [query]

    if q_sem and q_sem.lower() != query.lower():
        queries.append(q_sem)
    for mq in q_multi:
        if mq.lower() not in [x.lower() for x in queries]:
            queries.append(mq)
    
    print(
        f"[FALLBACK_EXPANSION] "
        f"queries={len(queries)} | "
        f"{' || '.join(q[:60] for q in queries)}"
    )

    # 2) retrieval ampio su più query e merge "grezzo" (dedup)
    all_hits = []
    seen = set()

    def _add_hits(hits):
        nonlocal all_hits, seen
        for h in hits:
            key = h.get("doc_id") or h.get("text", "")[:200]
            if key in seen:
                continue
            seen.add(key)
            all_hits.append(h)

    for q in queries:
        if search_technique == "dense":
            hits = dense_search(
                query=q,
                embedding_model=embedding_model,
                embedding_model_name=embedding_model_name,
                vectorstores=vectorstores,
                top_k=candidate_k,
                classification_mode="rag",
                use_reranking=False,  # IMPORTANT: non usare reranking normale qui
            )
        elif search_technique == "sparse":
            hits = bm25_search(corpus, q, bm25, nlp, k=candidate_k, classification_mode="rag")
        else:  # "hybrid"
            hits = hybrid_search(
                query=q,
                embedding_model=embedding_model,
                embedding_model_name=embedding_model_name,
                vectorstores=vectorstores,
                corpus=corpus,
                bm25=bm25,
                nlp=nlp,
                k=candidate_k,
                classification_mode="rag",
                use_reranking=False,
            )

        _add_hits(hits)

    # 3) deep rerank sui candidati complessivi
    # Nota: se sono troppi, taglio prima
    #all_hits = all_hits[:max(candidate_k, final_top_k)]
    all_hits = all_hits[:max(candidate_k * 3, final_top_k)]
    print(
        f"[DEEP_RERANK] "
        f"candidates_total={len(all_hits)} | "
        f"rerank_k={candidate_k} | "
        f"final_k={final_top_k}"
    )
    return deep_rerank_with_cross_encoder(query=query, hits=all_hits, reranker=reranker, top_k=final_top_k, rerank_k=candidate_k)
