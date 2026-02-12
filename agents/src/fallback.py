from prompts import QUERY_EXPANSION_PROMPT, MULTI_QUERY_PROMPT
from retrieval import dense_search
from reranking import rerank_with_cross_encoder
    

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
    embedding_model,
    embedding_model_name: str,
    vectorstores,
    llm=None,
    reranker=None,
    final_top_k: int = 5,
    candidate_k: int = 25,
) -> list[dict]:
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
        hits = dense_search(
            query=q,
            embedding_model=embedding_model,
            embedding_model_name=embedding_model_name,
            vectorstores=vectorstores,
            top_k=candidate_k,
            classification_mode="rag",
            use_reranking=False,
        )
        _add_hits(hits)

    print(
        f"[DEEP_RERANK] "
        f"candidates_total={len(all_hits)} | "
        f"final_k={final_top_k}"
    )
    return rerank_with_cross_encoder(query=query, docs=all_hits, reranker=reranker, top_k=final_top_k)