from prompts import EXAM_CALENDAR_PROMPT, GRADUATION_CALENDAR_PROMPT, MODULES_PROMPT, RAG_PROMPT, TIMETABLE_PROMPT, CLASSIFIER_PROMPT, simple_prompt_template, QUERY_REWRITE_PROMPT
from retrieval import bm25_search, dense_search, hybrid_search


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
