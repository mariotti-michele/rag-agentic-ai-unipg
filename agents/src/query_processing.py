from prompts import EXAM_CALENDAR_PROMPT, PROGRAM_REGULATIONS_PROMPT, RAG_PROMPT, TIMETABLE_PROMPT, CLASSIFIER_PROMPT, TIMETABLE_PROMPT, simple_prompt_template
from retrieval import bm25_search, dense_search, hybrid_search

def build_context(docs: list) -> str:
    context = ""
    for i, d in enumerate(docs, 1):
        section = f" | Sezione: {d.get('section_path','')}" if d.get("section_path") else ""
        context += f"[Fonte {i}] ({d.get('collection','N/A')}){section}\n{d['text']}\n\n"
    return context


def get_llm_answer(context: str, query: str, llm, prompt_template) -> str:
    answer = llm.invoke(prompt_template.format(context=context, question=query))
    if hasattr(answer, "content"):
        answer = answer.content
    return answer


def process_query(docs: list, query: str, llm, classification_mode) -> tuple[str, list]:
    if not docs:
        return "Non presente nei documenti", []
    context = build_context(docs)
    prompt_template = RAG_PROMPT
    if classification_mode == "orario":
        prompt_template = TIMETABLE_PROMPT
    elif classification_mode == "calendario esami":
        prompt_template = EXAM_CALENDAR_PROMPT
    elif classification_mode == "regolamenti":
        prompt_template = PROGRAM_REGULATIONS_PROMPT
    answer = get_llm_answer(context, query, llm, prompt_template)
    return answer, [d["text"] for d in docs]


def answer_query_dense(query: str, embedding_model, embedding_model_name: str, vectorstores, llm, classification_mode):
    dense_docs = dense_search(query, embedding_model, embedding_model_name, vectorstores, classification_mode=classification_mode)
    return process_query(dense_docs, query, llm, classification_mode)

def answer_query_bm25(query: str, corpus, bm25, nlp, llm, classification_mode):
    sparse_docs = bm25_search(corpus, query, bm25, nlp, classification_mode=classification_mode)
    return process_query(sparse_docs, query, llm, classification_mode)

def answer_query_hybrid(query: str, embedding_model, embedding_model_name, vectorstores, corpus, bm25, nlp, llm, classification_mode):
    merged_docs = hybrid_search(query, embedding_model, embedding_model_name, vectorstores, corpus, bm25, nlp, classification_mode=classification_mode)
    return process_query(merged_docs, query, llm, classification_mode)


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
        elif "regolamenti" in classification:
            print("[INFO] Query classificata come regolamenti.")
            return "regolamenti"
        else:
            return "rag"
    except Exception as e:
        print(f"[WARN] Errore classificazione query: {e}")
        return "rag"


def generate_answer(llm, query: str, search_technique, embedding_model, embedding_model_name, vectorstores, corpus, bm25, nlp):
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
            answer, contexts = answer_query_dense(query, embedding_model, embedding_model_name, vectorstores, llm, mode)
        elif search_technique == "sparse":
            answer, contexts = answer_query_bm25(query, corpus, bm25, nlp, llm, mode)
        elif search_technique == "hybrid":
            answer, contexts = answer_query_hybrid(query, embedding_model, embedding_model_name, vectorstores, corpus, bm25, nlp, llm, mode)
    return answer, contexts, mode
