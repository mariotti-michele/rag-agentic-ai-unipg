import numpy as np
# TF-IDF
# from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import spacy
from reranking import rerank_documents, rerank_with_cross_encoder
from prompts import QUERY_EXPANSION_PROMPT, MULTI_QUERY_PROMPT


def build_corpus(qdrant_client, collection_names) -> tuple[list[dict], list[str]]:
    corpus = []
    try:
        existing = qdrant_client.get_collections().collections
        existing_names = [c.name for c in existing]
    except Exception as e:
        print(f"[WARN] Impossibile leggere elenco collezioni da Qdrant: {e}")
        existing_names = []

    for collection_name in collection_names:
        if collection_name not in existing_names:
            continue

        next_offset = None
        total_texts = 0

        while True:
            try:
                points, next_page = qdrant_client.scroll(
                    collection_name=collection_name,
                    with_payload=True,
                    limit=1000,
                    offset=next_offset
                )
            except Exception as e:
                print(f"[WARN] Errore nel leggere {collection_name}: {e}")
                break

            if not points:
                break

            for p in points:
                text = p.payload.get("page_content", "")
                if text:
                    corpus.append({
                        "text": text,
                        "collection": collection_name,
                        "source_url": p.payload.get("source_url", ""),
                        "section_path": p.payload.get("section_path", ""),
                        "doc_id": p.payload.get("doc_id", ""),
                    })
                    total_texts += 1

            if next_page is None:
                break
            next_offset = next_page

        if total_texts == 0:
            print(f"[WARN] Nessun testo trovato in {collection_name}")

    if not corpus:
        raise RuntimeError("Nessun testo trovato in nessuna collezione!")

    print(f"[INFO] Corpus completo: {len(corpus)} chunk con metadata")

    all_texts = [c["text"] for c in corpus]

    return corpus, all_texts


# def build_tfidf(corpus_texts: list[str]):
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(corpus_texts)
#     return vectorizer, tfidf_matrix

# def tfidf_search_idx(query: str, k: int = 5):
#     qv = vectorizer.transform([query])
#     scores = (tfidf_matrix @ qv.T).toarray().ravel()
#     top_idx = np.argsort(scores)[::-1][:k]
#     return [(i, float(scores[i])) for i in top_idx]


def build_spacy_tokenizer():
    nlp = spacy.load("it_core_news_sm", disable=["parser", "ner", "tagger", "lemmatizer"])
    # nlp = spacy.load("it_core_news_sm", disable=["parser", "ner"])
    return nlp

def spacy_tokenize(text: str, nlp):
    doc = nlp(text.lower())
    return [t.text for t in doc if not t.is_space and not t.is_punct and not t.is_stop]
    # return [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_space]


def build_bm25(corpus_texts: list[str], nlp):
    tokenized_corpus = [spacy_tokenize(t, nlp) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def bm25_search_idx(query: str, bm25, nlp, k: int = 5):
    qtoks = spacy_tokenize(query, nlp)
    scores = bm25.get_scores(qtoks)
    top_idx = np.argsort(scores)[::-1][:k]
    return [(i, float(scores[i])) for i in top_idx]

def bm25_search(corpus, query: str, bm25, nlp, k: int = 5, classification_mode="rag"):
    collection_filter_value = collection_filter(classification_mode) if classification_mode != "rag" else None
    sparse_idxs = bm25_search_idx(query, bm25, nlp, k=k*2 if collection_filter_value else k)
    
    sparse_docs = []
    for i, score in sparse_idxs:
        if collection_filter_value and corpus[i]["collection"] != collection_filter_value:
            continue
        sparse_docs.append({
            **corpus[i],
            "score": score,
        })
        if len(sparse_docs) >= k:
            break
    
    return sparse_docs


def collection_filter(classification_mode: str):
    if classification_mode == "orario":
        return "ing_info_mag_orari"
    elif classification_mode == "calendario esami":
        return "ing_info_mag_calendario_esami"
    elif classification_mode == "insegnamenti":
        return ["ing_info_mag_regolamenti_didattici_tabelle", "ing_info_mag_regolamenti_didattici"]
    elif classification_mode == "calendario lauree":
        return "ing_info_calendario_lauree"
    else:
        return None


def dense_search(query: str, embedding_model, embedding_model_name: str, vectorstores, top_k: int = 5, classification_mode="rag", use_reranking=False, llm=None, rerank_method="cross_encoder", reranker=None):
    if embedding_model_name == "e5" or embedding_model_name == "bge":
        query = "query: " + query

    collection_filter_value = None
    if classification_mode != "rag":
        collection_filter_value = collection_filter(classification_mode)

    vec = embedding_model.embed_query(query)
    hits = []

    # Recupera più documenti se si usa il re-ranking
    search_k = top_k * 2 if use_reranking else top_k

    for name, store in vectorstores.items():
        if collection_filter_value:
            if isinstance(collection_filter_value, list):
                if name not in collection_filter_value:
                    continue
            elif name != collection_filter_value:
                continue
        try:
            docs_scores = store.similarity_search_with_score_by_vector(vec, k=search_k)

            for doc, score in docs_scores:
                hits.append({
                    "text": doc.page_content,
                    "collection": name,
                    "source_url": doc.metadata.get("source_url", ""),
                    "section_path": doc.metadata.get("section_path", ""),
                    "doc_id": doc.metadata.get("doc_id", ""),
                    "score": float(score),
                })
        except Exception as e:
            print(f"[WARN] Errore su {name}: {e}")

    hits.sort(key=lambda x: x["score"], reverse=True)
    
    # Applica re-ranking se richiesto
    if use_reranking and len(hits) > top_k:
        hits_to_rerank = hits[:search_k]
        
        if rerank_method == "llm" and llm:
            hits = rerank_documents(query, hits_to_rerank, llm, top_k)
        elif rerank_method == "cross_encoder":
            hits = rerank_with_cross_encoder(query, hits_to_rerank, reranker=reranker, top_k=top_k)
        else:
            hits = hits[:top_k]
    else:
        hits = hits[:top_k]
    
    return hits


def reciprocal_rank_fusion_docs(dense_docs, sparse_docs, alpha=60, k=5):
    combined = {}
    for rank, d in enumerate(dense_docs):
        docid = d.get("doc_id") if d.get("doc_id") else d["text"]
        combined[docid] = combined.get(docid, 0) + 1 / (alpha + rank + 1)
    for rank, d in enumerate(sparse_docs):
        docid = d.get("doc_id") if d.get("doc_id") else d["text"]
        combined[docid] = combined.get(docid, 0) + 1 / (alpha + rank + 1)

    ranked_ids = [rid for rid, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]]
    merged = {(d.get("doc_id") if d.get("doc_id") else d["text"]): d for d in dense_docs + sparse_docs}
    return [merged[rid] for rid in ranked_ids if rid in merged]


def hybrid_search(query, embedding_model, embedding_model_name, vectorstores, corpus, bm25, nlp, alpha=60, k=5, classification_mode="rag", use_reranking=False, llm=None, rerank_method="cross_encoder", reranker=None):
    dense_docs = dense_search(query, embedding_model, embedding_model_name, vectorstores, top_k=k*2, classification_mode=classification_mode)
    sparse_docs = bm25_search(corpus, query, bm25, nlp, k=k*2, classification_mode=classification_mode)

    if not dense_docs and not sparse_docs:
        return []

    target_merge_size = k*2 if use_reranking else k
    merged_docs = reciprocal_rank_fusion_docs(dense_docs, sparse_docs, alpha=alpha, k=target_merge_size)
    
    # Applica re-ranking sui documenti fusi
    if use_reranking and len(merged_docs) > k:
        
        if rerank_method == "llm" and llm:
            merged_docs = rerank_documents(query, merged_docs, llm, k)
        elif rerank_method == "cross_encoder":
            merged_docs = rerank_with_cross_encoder(query, merged_docs, reranker=reranker, top_k=k)
        else:
            merged_docs = merged_docs[:k]
    
    return merged_docs[:k]


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
