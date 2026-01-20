import argparse

from initializer import init_components, test_connection
from query_processing import generate_answer
from retrieval import build_bm25, build_corpus, build_spacy_tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Sistema Q&A con modelli selezionabili")
    parser.add_argument("--llm-model", type=str, default="gemini",
                        choices=["llama-local", "gemini", "llama-api"],
                        help="Seleziona il modello da usare")
    parser.add_argument("--embedding-model", type=str, default="nomic",
                        choices=["nomic", "e5", "all-mpnet"],
                        help="Seleziona il modello di embedding da usare")
    parser.add_argument("--search", type=str, default="hybrid",
                        choices=["dense", "sparse", "hybrid"],
                        help="Seleziona tecnica di ricerca da utilizzare (default: hybrid)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    llm_model_name, embedding_model_name, search_technique = args.llm_model, args.embedding_model, args.search

    embedding_model, vectorstores, llm, COLLECTION_NAMES, qdrant_client = init_components(embedding_model_name=embedding_model_name, llm_model_name=llm_model_name)
    corpus, corpus_texts = build_corpus(qdrant_client, COLLECTION_NAMES)
    spacy_tokenizer = build_spacy_tokenizer()
    bm25 = build_bm25(corpus_texts, spacy_tokenizer)
    
    print("Sistema di Q&A avviato.")

    if not test_connection(vectorstores, embedding_model):
        print("Impossibile connettersi al vector store. Verificare che la collezione esista.")
        exit(1)

    print("Connessione verificata. Digita 'exit' o 'quit' per uscire.\n")

    while True:
        try:
            q = input("Domanda: ")
            if q.lower() in ["exit", "quit"]:
                break
            if q.strip():
                answer, contexts, mode = generate_answer(llm, q, search_technique, embedding_model, embedding_model_name, vectorstores, corpus, bm25, spacy_tokenizer)
                if(mode == "semplice"):
                    print(f"\nRisposta semplice:\n")
                else:
                    print("\nRisposta sistema RAG:\n")
                print(answer)
                print("-" * 50)
            else:
                print("Inserisci una domanda valida.")
        except KeyboardInterrupt:
            print("\nUscita...")
            break
        except Exception as e:
            print(f"Errore: {e}")