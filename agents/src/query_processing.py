from prompts import EXAM_CALENDAR_PROMPT, GRADUATION_CALENDAR_PROMPT, MODULES_PROMPT, RAG_PROMPT, TIMETABLE_PROMPT, CLASSIFIER_PROMPT, QUERY_REWRITE_PROMPT, QUESTION_DECOMPOSITION_PROMPT, ANSWER_COMBINATION_PROMPT
from urllib.parse import urlparse, unquote
import os


def build_context(docs: list) -> str:
    context = ""
    for i, d in enumerate(docs, 1):
        section = f" | Sezione: {d.get('section_path','')}" if d.get("section_path") else ""
        context += f"[Fonte {i}] ({d.get('collection','N/A')}){section}\n{d['text']}\n\n"
    return context

def _title_from_url(url: str) -> str:
    try:
        path = unquote(urlparse(url).path)
        name = os.path.basename(path)
        return name or url
    except Exception:
        return url


def build_references(docs: list[dict]) -> list[dict]:
    refs = []
    seen = set()

    for d in docs:
        url = d.get("source_url") or ""
        if not url:
            continue

        title = _title_from_url(url)

        section = d.get("section_path") or ""
        
        if not title or title == url:
            doc_id = d.get("doc_id") or ""
            if doc_id:
                title = doc_id

        key = (url, title, section)
        if key in seen:
            continue
        seen.add(key)

        refs.append({"url": url, "title": title, "section": section})

    return refs


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

    #print(f"[DEBUG] Prompt per LLM:\n{prompt}\n")
    answer = llm.invoke(prompt)
    if hasattr(answer, "content"):
        answer = answer.content
    return answer


def process_query(docs: list, query: str, llm, classification_mode, memory_context: str = "") -> tuple[str, list]:
    if not docs:
        return "Non presente nei documenti", []
    context = build_context(docs)
    #print("\n[DEBUG] ===== CONTEXT PASSATO ALL'LLM =====")
    #print(context)
    #print("[DEBUG] ===== FINE CONTEXT =====\n")
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
    references = build_references(docs)
    #return answer, [d["text"] for d in docs]
    return answer, references


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
