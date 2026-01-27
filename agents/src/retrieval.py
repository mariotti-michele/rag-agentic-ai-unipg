import numpy as np
# TF-IDF
# from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import spacy


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
        return "ing_info_mag_regolamenti_didattici_tabelle"
    else:
        return None


def dense_search(query: str, embedding_model, embedding_model_name: str, vectorstores, top_k: int = 5, classification_mode="rag", use_reranking=False, llm=None, rerank_method="cross_encoder"):
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
        if collection_filter_value and name != collection_filter_value:
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
        from advanced_techniques import rerank_documents, rerank_with_cross_encoder
        
        if rerank_method == "llm" and llm:
            hits = rerank_documents(query, hits, llm, top_k)
        elif rerank_method == "cross_encoder":
            hits = rerank_with_cross_encoder(query, hits, top_k=top_k)
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


def hybrid_search(query, embedding_model, embedding_model_name, vectorstores, corpus, bm25, nlp, alpha=60, k=5, classification_mode="rag", use_reranking=False, llm=None, rerank_method="cross_encoder"):
    dense_docs = dense_search(query, embedding_model, embedding_model_name, vectorstores, top_k=k*2, classification_mode=classification_mode)
    sparse_docs = bm25_search(corpus, query, bm25, nlp, k=k*2, classification_mode=classification_mode)

    if not dense_docs and not sparse_docs:
        return []

    merged_docs = reciprocal_rank_fusion_docs(dense_docs, sparse_docs, alpha=alpha, k=k*2 if use_reranking else k)
    
    # Applica re-ranking sui documenti fusi
    if use_reranking and len(merged_docs) > k:
        from advanced_techniques import rerank_documents, rerank_with_cross_encoder
        
        if rerank_method == "llm" and llm:
            merged_docs = rerank_documents(query, merged_docs, llm, k)
        elif rerank_method == "cross_encoder":
            merged_docs = rerank_with_cross_encoder(query, merged_docs, top_k=k)
        else:
            merged_docs = merged_docs[:k]
    
    return merged_docs[:k]
