import argparse

from initializer import init_components, test_connection
from retrieval import build_bm25, build_corpus, build_spacy_tokenizer
from conversation_memory import ConversationMemory
from rag_graph import build_rag_graph

def parse_args():
    parser = argparse.ArgumentParser(description="Sistema Q&A con modelli selezionabili")
    parser.add_argument("--llm-model", type=str, default="vllm",
                        choices=["llama-local", "gemini", "llama-api", "vllm"],
                        help="Seleziona il modello da usare")
    parser.add_argument("--embedding-model", type=str, default="bge",
                        choices=["nomic", "e5", "all-mpnet", "bge"],
                        help="Seleziona il modello di embedding da usare")
    parser.add_argument("--search", type=str, default="dense",
                        choices=["dense", "sparse", "hybrid"],
                        help="Seleziona tecnica di ricerca da utilizzare (default: dense)")
    parser.add_argument("--reranking", action="store_true",
                        help="Attiva il re-ranking dei documenti")
    parser.add_argument("--rerank-method", type=str, default="cross_encoder",
                        choices=["cross_encoder", "llm"],
                        help="Metodo di re-ranking: cross_encoder (veloce) o llm (accurato)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    llm_model_name, embedding_model_name, search_technique = args.llm_model, args.embedding_model, args.search
    use_reranking = args.reranking
    rerank_method = args.rerank_method

    embedding_model, vectorstores, llm, COLLECTION_NAMES, qdrant_client, reranker = init_components(embedding_model_name=embedding_model_name, llm_model_name=llm_model_name)
    corpus, corpus_texts = build_corpus(qdrant_client, COLLECTION_NAMES)
    spacy_tokenizer = build_spacy_tokenizer()
    bm25 = build_bm25(corpus_texts, spacy_tokenizer)
    
    # Crea il grafo RAG
    rag_graph = build_rag_graph()
    
    # Inizializza la memoria conversazionale
    memory = ConversationMemory(max_turns=4)
    
    print("Sistema di Q&A avviato.")
    print(f"Tecnica di ricerca: {search_technique}")
    if use_reranking:
        print(f"Re-ranking attivo: metodo {rerank_method}")
        if rerank_method == "cross_encoder" and reranker:
            print("[INFO] Reranker BGE API disponibile")
        elif rerank_method == "cross_encoder" and not reranker:
            print("[WARN] Reranker BGE API non disponibile, verranno usati scores originali")

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
                # Ottieni il contesto dalla memoria
                memory_context = memory.get_context()
                
                # Invoca il grafo
                result = rag_graph.invoke({
                    "question": q,
                    "session_id": "cli_session",
                    "memory_context": memory_context,
                    "search_technique": search_technique,
                    "llm": llm,
                    "embedding_model": embedding_model,
                    "embedding_model_name": embedding_model_name,
                    "vectorstores": vectorstores,
                    "corpus": corpus,
                    "bm25": bm25,
                    "nlp": spacy_tokenizer,
                    "use_reranking": use_reranking,
                    "rerank_method": rerank_method,
                    "reranker": reranker,
                })
                
                answer = result["answer"]
                mode = result.get("mode", "rag")
                
                # Aggiungi alla memoria
                memory.add_turn(q, answer)
                
                if mode == "semplice":
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