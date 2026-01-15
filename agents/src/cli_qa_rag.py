import argparse

parser = argparse.ArgumentParser(description="Sistema Q&A con modelli selezionabili")
parser.add_argument("--model", type=str, default="gemini",
                    choices=["llama-local", "gemini", "llama-api"],
                    help="Seleziona il modello da usare")
args = parser.parse_args()

# def answer_query_dense(query: str, k: int = 5):
#     dense_docs = dense_search(query, top_k=k)
#     if not dense_docs:
#         return "Non presente nei documenti", []

#     context = ""
#     for i, d in enumerate(dense_docs, 1):
#         section = f" | Sezione: {d.get('section_path','')}" if d.get("section_path") else ""
#         context += f"[Fonte {i}] ({d.get('collection','N/A')}){section}\n{d.get('text','')}\n\n"

#     prompt = f"""{QA_CHAIN_PROMPT.format(context=context, question=query)}

# Rispondi in un unico paragrafo chiaro e completo, senza aggiungere sezioni o titoli.
# """
#     answer = llm.invoke(prompt)
#     if hasattr(answer, "content"):
#         answer = answer.content
#     return f"Risposta Dense: {answer}", [d["text"] for d in dense_docs]


# def answer_query_bm25(query: str, k: int = 5):
#     sparse_idxs = bm25_search_idx(query, k=k)
#     if not sparse_idxs:
#         return "Non presente nei documenti", []

#     sparse_docs = [{**corpus[i], "score": score} for i, score in sparse_idxs]

#     context = ""
#     for i, d in enumerate(sparse_docs, 1):
#         section = f" | Sezione: {d.get('section_path','')}" if d.get("section_path") else ""
#         context += f"[Fonte {i}] ({d.get('collection','N/A')}){section}\n{d['text']}\n\n"

#     prompt = f"""{QA_CHAIN_PROMPT.format(context=context, question=query)}

# Rispondi in un unico paragrafo chiaro e completo, senza aggiungere sezioni o titoli.
# """
#     answer = llm.invoke(prompt)
#     if hasattr(answer, "content"):
#         answer = answer.content

#     return f"Risposta Sparse (BM25): {answer}", [d["text"] for d in sparse_docs]


def classify_query(query: str) -> str:
    try:
        classification = llm.invoke(classifier_prompt.format(question=query))
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


if __name__ == "__main__":
    print("Sistema di Q&A avviato.")

    if not test_connection():
        print("Impossibile connettersi al vector store. Verificare che la collezione esista.")
        exit(1)

    print("Connessione verificata. Digita 'exit' o 'quit' per uscire.\n")

    while True:
        try:
            q = input("Domanda: ")
            if q.lower() in ["exit", "quit"]:
                break
            if q.strip():
                mode = classify_query(q)
                if mode == "semplice":
                    prompt = simple_prompt_template.format(question=q)
                    answer = llm.invoke(prompt)
                    if hasattr(answer, "content"):
                        answer = answer.content
                    print(f"\nRisposta semplice: {answer}\n")
                else:
                    print("\n--- Risposta Ibrida (Dense + BM25) ---")
                    hybrid_answer, contexts = hybrid_search(q)
                    print(hybrid_answer)

                print("-" * 50)
            else:
                print("Inserisci una domanda valida.")
        except KeyboardInterrupt:
            print("\nUscita...")
            break
        except Exception as e:
            print(f"Errore: {e}")