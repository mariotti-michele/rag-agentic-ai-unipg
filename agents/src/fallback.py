from prompts import QUERY_EXPANSION_PROMPT, MULTI_QUERY_PROMPT
from retrieval import dense_search, bm25_search, hybrid_search

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
    

def semantic_query_expansion(llm, query: str) -> str:
    if not llm:
        return query
    out = llm.invoke(QUERY_EXPANSION_PROMPT.format(query=query))
    if hasattr(out, "content"):
        out = out.content
    q2 = str(out).strip().strip('"')
    return q2 if q2 else query


def multi_query_expansion(llm, query: str, n: int = 3) -> list[str]:
    if not llm:
        return [query]
    out = llm.invoke(MULTI_QUERY_PROMPT.format(query=query))
    if hasattr(out, "content"):
        out = out.content
    lines = [l.strip().strip('"') for l in str(out).splitlines() if l.strip()]
    qs = []
    for l in lines:
        if l and l.lower() != query.lower():
            qs.append(l)
    if not qs:
        qs = [query]
    return qs[:n]


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
        f"{' || '.join(q for q in queries)}"
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
