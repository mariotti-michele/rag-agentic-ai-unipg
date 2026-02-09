import json
from pathlib import Path
import sys
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from initializer import init_components
from retrieval import build_bm25, build_corpus, build_spacy_tokenizer
from rag_graph import build_rag_graph


def parse_args():
    parser = argparse.ArgumentParser(description="Valutazione manuale RAG")
    parser.add_argument("--llm-model", type=str, default="vllm",
                        choices=["llama-local", "gemini", "llama-api", "vllm"],
                        help="Seleziona il modello da usare")
    parser.add_argument("--embedding-model", type=str, default="bge",
                        choices=["nomic", "e5", "all-mpnet", "bge"],
                        help="Seleziona il modello di embedding da usare")
    parser.add_argument("--search", type=str, default="dense",
                        choices=["all", "dense", "sparse", "hybrid"],
                        help="Seleziona tecnica di ricerca (default: dense)")
    parser.add_argument("--reranking", action="store_true",
                        help="Attiva il re-ranking dei documenti")
    parser.add_argument("--rerank-method", type=str, default="cross_encoder",
                        choices=["cross_encoder", "llm"],
                        help="Metodo di re-ranking: cross_encoder (veloce) o llm (accurato)")
    args = parser.parse_args()
    return args


def run_manual_eval(embedding_model, embedding_model_name, vectorstores, llm, corpus, bm25, nlp, reranker, search_technique, use_reranking=False, rerank_method="cross_encoder"):
    print("="*60)
    print("CONFIGURAZIONE VALUTAZIONE MANUALE")
    print("="*60)
    print(f"Tecnica di ricerca: {search_technique}")
    print(f"Re-ranking attivo: {'Sì' if use_reranking else 'No'}")
    if use_reranking:
        print(f"Metodo re-ranking: {rerank_method}")
    print("="*60 + "\n")
    
    # Crea il grafo RAG
    rag_graph = build_rag_graph()
    
    VALIDATION_DIR = Path(__file__).resolve().parent / "validation_set"   # human_validation_set / validation_set
    print(f"Caricamento dataset da: {VALIDATION_DIR}")

    validation_data = []
    for json_file in sorted(VALIDATION_DIR.glob("*.json")):
        print(f"  -> Trovato file: {json_file.name}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    validation_data.extend(content)
                else:
                    print(f"Il file {json_file.name} non contiene una lista JSON valida, ignorato.")
        except Exception as e:
            print(f"Errore nel file {json_file.name}: {e}")

    print(f"Totale domande caricate: {len(validation_data)}\n")

    for i, d in enumerate(validation_data, start=1):
        q = d["question"]
        print("-"*50)
        print(f"[{i}] Domanda: {q}\n")

        try:
            if search_technique == "all":
                # Esegui tutte e tre le tecniche
                for technique in ["dense", "sparse", "hybrid"]:
                    result = rag_graph.invoke({
                        "question": q,
                        "memory_context": "",
                        "search_technique": technique,
                        "llm": llm,
                        "embedding_model": embedding_model,
                        "embedding_model_name": embedding_model_name,
                        "vectorstores": vectorstores,
                        "corpus": corpus,
                        "bm25": bm25,
                        "nlp": nlp,
                        "use_reranking": use_reranking,
                        "rerank_method": rerank_method,
                        "reranker": reranker,
                    })
                    response = result["answer"]
                    retrieved_ctx = result.get("contexts", [])
                    print(f"Risposta ({technique}):\n{response}\n")
            else:
                # Esegui solo la tecnica specificata
                result = rag_graph.invoke({
                    "question": q,
                    "memory_context": "",
                    "search_technique": search_technique,
                    "llm": llm,
                    "embedding_model": embedding_model,
                    "embedding_model_name": embedding_model_name,
                    "vectorstores": vectorstores,
                    "corpus": corpus,
                    "bm25": bm25,
                    "nlp": nlp,
                    "use_reranking": use_reranking,
                    "rerank_method": rerank_method,
                    "reranker": reranker,
                })
                response = result["answer"]
                retrieved_ctx = result.get("contexts", [])
                print(f"Risposta ({search_technique}):\n{response}\n")

            print("Contesti recuperati:")
            for c in retrieved_ctx:
                print("-", c[:200], "...")
        except Exception as e:
            print(f"Errore durante la domanda '{q}': {e}")
            import traceback
            traceback.print_exc()
            print("Risposta generata: [ERRORE]\n")

    print("-"*50)
    print("Completato. Tutte le risposte sono state stampate.")


if __name__ == "__main__":
    args = parse_args()
    llm_model_name, embedding_model_name, search_technique = args.llm_model, args.embedding_model, args.search
    use_reranking = args.reranking
    rerank_method = args.rerank_method

    embedding_model, vectorstores, llm, COLLECTION_NAMES, qdrant_client, reranker = init_components(
        embedding_model_name=embedding_model_name,
        llm_model_name=llm_model_name
    )
    corpus, corpus_texts = build_corpus(qdrant_client, COLLECTION_NAMES)
    spacy_tokenizer = build_spacy_tokenizer()
    bm25 = build_bm25(corpus_texts, spacy_tokenizer)

    run_manual_eval(embedding_model, embedding_model_name, vectorstores, llm, corpus, bm25, spacy_tokenizer, reranker, search_technique, use_reranking, rerank_method)
