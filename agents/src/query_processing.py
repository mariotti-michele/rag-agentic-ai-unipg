from prompts import QA_CHAIN_PROMPT, CLASSIFIER_PROMPT, simple_prompt_template
from retrieval import bm25_search_idx, dense_search, hybrid_search

def build_context(docs: list) -> str:
    context = ""
    for i, d in enumerate(docs, 1):
        section = f" | Sezione: {d.get('section_path','')}" if d.get("section_path") else ""
        context += f"[Fonte {i}] ({d.get('collection','N/A')}){section}\n{d['text']}\n\n"
    return context


def get_llm_answer(context: str, query: str, llm) -> str:
    answer = llm.invoke(QA_CHAIN_PROMPT.format(context=context, question=query))
    if hasattr(answer, "content"):
        answer = answer.content
    return answer


def process_query(docs: list, query: str, llm) -> tuple[str, list]:
    if not docs:
        return "Non presente nei documenti", []
    context = build_context(docs)
    answer = get_llm_answer(context, query, llm)
    return answer, [d["text"] for d in docs]


def answer_query_dense(query: str, embedding_model, vectorstores, llm):
    dense_docs = dense_search(query, embedding_model, vectorstores)
    return process_query(dense_docs, query, llm)

def answer_query_bm25(query: str, corpus, bm25, nlp, llm):
    sparse_idxs = bm25_search_idx(query, bm25, nlp)
    sparse_docs = [{**corpus[i], "score": score} for i, score in sparse_idxs] if sparse_idxs else []
    return process_query(sparse_docs, query, llm)

def answer_query_hybrid(query: str, embedding_model, vectorstores, corpus, bm25, nlp, llm):
    merged_docs = hybrid_search(query, embedding_model, vectorstores, corpus, bm25, nlp)
    return process_query(merged_docs, query, llm)


def classify_query(llm, query: str) -> str:
    try:
        classification = llm.invoke(CLASSIFIER_PROMPT.format(question=query))
        if hasattr(classification, "content"):
            classification = classification.content
        classification = str(classification).strip().lower()

        if "semplice" in classification:
            return "semplice"
        else:
            return "rag"
    except Exception as e:
        print(f"[WARN] Errore classificazione query: {e}")
        return "rag"

def generate_answer(llm, query: str, search_technique, embedding_model, vectorstores, corpus, bm25, nlp):
    mode = classify_query(llm, query)
    if mode == "semplice":
        prompt = simple_prompt_template.format(question=query)
        answer = llm.invoke(prompt)
        if hasattr(answer, "content"):
            answer = answer.content
        return answer, [], mode
    else:
        answer, contexts = None, []
        if search_technique == "dense":
            answer, contexts = answer_query_dense(query, embedding_model, vectorstores, llm)
        elif search_technique == "sparse":
            answer, contexts = answer_query_bm25(query, corpus, bm25, nlp, llm)
        elif search_technique == "hybrid":
            answer, contexts = answer_query_hybrid(query, embedding_model, vectorstores, corpus, bm25, nlp, llm)
    return answer, contexts, mode
