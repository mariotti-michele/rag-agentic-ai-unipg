import json
import os
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Callable, Tuple, List
import pandas as pd

# Disabilita LangSmith tracing PRIMA di importare altre librerie
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"] = ""

from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.metrics._answer_relevance import answer_relevancy
from ragas import evaluate
from ragas.run_config import RunConfig

from langsmith import Client

sys.path.append(str(Path(__file__).parent.parent))

from initializer import init_components
from retrieval import build_bm25, build_corpus, build_spacy_tokenizer
from rag_graph import build_rag_graph

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Valutazione RAG automatica con RAGAS")
    parser.add_argument("--llm-model", type=str, default="vllm",
                        choices=["llama-local", "gemini", "llama-api", "vllm"],
                        help="Seleziona il modello da usare")
    parser.add_argument("--embedding-model", type=str, default="bge",
                        choices=["nomic", "e5", "all-mpnet", "bge"],
                        help="Seleziona il modello di embedding da usare")
    parser.add_argument("--search", type=str, default="dense",
                        choices=["dense", "sparse", "hybrid", "all"],
                        help="Seleziona tecnica di ricerca da utilizzare (default: dense)")
    parser.add_argument("--chunking", type=str, default="section-limited",
                        choices=["fixed", "document-structure", "section", "semantic", "section-limited"],
                        help="Tipo di chunking usato per creare la collezione (default: section-limited)")
    parser.add_argument("--version", type=str, default="v0",
                        help="Versione del modello valutato (default: v0)")
    parser.add_argument("--reranking", action="store_true",
                        help="Attiva il re-ranking dei documenti")
    parser.add_argument("--rerank-method", type=str, default="cross_encoder",
                        choices=["cross_encoder", "llm"],
                        help="Metodo di re-ranking: cross_encoder (veloce) o llm (accurato)")
    args = parser.parse_args()
    return args


def retry_with_backoff(func: Callable, max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Esegue una funzione con retry e backoff esponenziale in caso di errore.
    
    Args:
        func: Funzione da eseguire
        max_retries: Numero massimo di tentativi
        initial_delay: Ritardo iniziale in secondi
        backoff_factor: Fattore di moltiplicazione del ritardo ad ogni retry
    
    Returns:
        Il risultato della funzione se ha successo
    
    Raises:
        L'ultima eccezione se tutti i retry falliscono
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_msg = str(e)
            
            # Log dell'errore
            print(f"   [Tentativo {attempt + 1}/{max_retries}] Errore: {error_msg}")
            
            # Se è l'ultimo tentativo, rilancia l'eccezione
            if attempt == max_retries - 1:
                print(f"   Tutti i {max_retries} tentativi falliti.")
                raise last_exception
            
            # Attendi prima del prossimo tentativo
            print(f"   Attendo {delay:.1f}s prima del prossimo tentativo...")
            time.sleep(delay)
            delay *= backoff_factor
    
    # Questo punto non dovrebbe mai essere raggiunto, ma per sicurezza
    raise last_exception


def evaluate_variant(rag_graph, embedding_model, vectorstores, corpus, bm25, nlp, reranker, llm, embedding_model_name: str, llm_model_name: str, chunking: str, search: str, version: str, use_reranking: bool = False, rerank_method: str = "cross_encoder"):
    rerank_suffix = f"_rerank_{rerank_method}" if use_reranking else ""
    
    print("\n" + "="*70)
    print("CONFIGURAZIONE VALUTAZIONE")
    print("="*70)
    print(f"LLM model: {llm_model_name}")
    print(f"Embedding model: {embedding_model_name}")
    print(f"Chunking: {chunking}")
    print(f"Search: {search}")
    print(f"Version: {version}")
    print(f"Re-ranking attivo: {'Sì' if use_reranking else 'No'}")
    if use_reranking:
        print(f"Metodo re-ranking: {rerank_method}")
    print("="*70 + "\n")

    base_dir = Path("evaluations_results") / llm_model_name / embedding_model_name / chunking / (search + f"_{version}" + rerank_suffix)
    base_dir.mkdir(parents=True, exist_ok=True)
    name = f"{llm_model_name}-{embedding_model_name}-{chunking}-{search}-{version}{rerank_suffix}"
    csv_path = base_dir / f"eval_results_{name.replace(' ', '_')}.csv"

    VALIDATION_DIR = Path(__file__).resolve().parent / "validation_set"
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

    print(f"Totale domande caricate: {len(validation_data)}")

    questions = [d["question"] for d in validation_data]
    ground_truths = [d.get("ground_truth", "") for d in validation_data]

    answers, retrieved_contexts, errors = [], [], []

    print(f"\nGenerazione risposte con variante {name} ({len(questions)} domande)...")
    for i, q in enumerate(questions, start=1):
        print(f" -> [{i}/{len(questions)}] {q}")
        try:
            # Usa il grafo per generare la risposta
            def generate_with_graph():
                result = rag_graph.invoke({
                    "question": q,
                    "memory_context": "",
                    "search_technique": search,
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
                return result["answer"], result.get("contexts", [])
            
            response, ctxs = retry_with_backoff(
                generate_with_graph,
                max_retries=3,
                initial_delay=2.0,
                backoff_factor=2.0
            )
            answers.append(response)
            retrieved_contexts.append(ctxs)
            errors.append("")  # Nessun errore
        except Exception as e:
            print(f"Errore permanente durante la domanda '{q}': {e}")
            answers.append("")
            retrieved_contexts.append([])
            errors.append(str(e))

    dataset = Dataset.from_dict({
        "question": questions,
        "contexts": retrieved_contexts,
        "answer": answers,
        "ground_truth": ground_truths
    })

    print(f"\nValutazione Ragas per {name}...")
    metrics_list = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]

    run_config = RunConfig(
        timeout=600,  # Aumentato timeout a 10 minuti
        max_workers=1,  # Ridotto a 1 worker per evitare overload
        max_wait=720  # Aumentato tempo massimo di attesa
    )

    # Aggiungi retry anche per la valutazione RAGAS
    def evaluate_with_retry():
        return evaluate(
            dataset=dataset, 
            metrics=metrics_list, 
            llm=llm, 
            embeddings=embedding_model,
            run_config=run_config
        )
    
    try:
        result = retry_with_backoff(
            evaluate_with_retry,
            max_retries=2,
            initial_delay=5.0,
            backoff_factor=2.0
        )
        result_df = result.to_pandas()
        
        # Calcola le medie in due modi
        results_with_nan = result_df.mean(numeric_only=True).to_dict()
        
        # Seconda media: solo righe senza errori, NaN = 0
        result_df_no_errors = result_df.copy()
        # Aggiungi colonna errori per filtrare
        result_df_no_errors['has_error'] = [bool(err) for err in errors]
        result_df_clean = result_df_no_errors[~result_df_no_errors['has_error']]
        result_df_clean = result_df_clean.fillna(0)  # NaN come 0
        results_clean = result_df_clean.drop(columns=['has_error']).mean(numeric_only=True).to_dict()

        print("\nRISULTATI RAGAS - MEDIE GLOBALI (esclude NaN):")
        for k, v in results_with_nan.items():
            print(f" - {k}: {v:.3f}")
        
        print("\nRISULTATI RAGAS - MEDIE CLEAN (solo righe senza errori, NaN=0):")
        for k, v in results_clean.items():
            print(f" - {k}: {v:.3f}")

        save_results_to_csv(csv_path, dataset, result_df, results_with_nan, results_clean, errors, search_technique=search)
        print(f"Risultati salvati in {csv_path}")
    except Exception as e:
        print(f"Errore permanente durante la valutazione RAGAS: {e}")
        print("Salvataggio risultati parziali...")
        # Salva comunque quello che hai ottenuto finora
        partial_csv = base_dir / f"partial_results_{name.replace(' ', '_')}.csv"
        with open(partial_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer", "contexts", "error"])
            for i in range(len(questions)):
                writer.writerow([questions[i], answers[i], str(retrieved_contexts[i]), errors[i]])
        print(f"Risultati parziali salvati in {partial_csv}")

    langsmith_key = os.getenv("LANGCHAIN_API_KEY")
    if langsmith_key and False: # Disabilitato temporaneamente
        try:
            client = Client()
            client.create_run(
                name=f"Evaluation RAG UNIPG {name}",
                run_type="chain",
                inputs={"questions": questions},
                outputs={"results": dict(results_with_nan)},
                 metadata={
                    "component": f"ragas_{name.lower().replace(' ', '_')}_evaluation",
                    "variant": name,
                    "version": version,
                    "num_questions": len(questions)
                },
                start_time=datetime.now(timezone.utc).isoformat(),
                end_time=datetime.now(timezone.utc).isoformat(),
                status="completed"
            )
            print(f"Risultati {name} inviati a LangSmith.")
        except Exception as e:
            print(f"[WARN] Errore durante l'invio a LangSmith: {e}")
    else:
        print("Nessuna API key LangSmith trovata — risultati solo in locale.")


def save_results_to_csv(csv_path: Path, dataset: Dataset, result_df, metrics_with_nan: dict, metrics_clean: dict, errors: list, search_technique: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Ricerca " + search_technique])
            writer.writerow([
                "question", "answer", "faithfulness",
                "answer_relevancy", "context_precision", "context_recall",
                "answer_correctness", "error"
            ])

        for i in range(len(dataset)):
            row = result_df.iloc[i]
            writer.writerow([
                dataset[i]["question"],
                dataset[i]["answer"],
                f"{row['faithfulness']:.3f}" if not pd.isna(row['faithfulness']) else "NaN",
                f"{row['answer_relevancy']:.3f}" if not pd.isna(row['answer_relevancy']) else "NaN",
                f"{row['context_precision']:.3f}" if not pd.isna(row['context_precision']) else "NaN",
                f"{row['context_recall']:.3f}" if not pd.isna(row['context_recall']) else "NaN",
                f"{row['answer_correctness']:.3f}" if not pd.isna(row['answer_correctness']) else "NaN",
                errors[i]
            ])

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            "--- MEDIA GLOBALE (esclude NaN) ---",
            f"{metrics_with_nan['faithfulness']:.3f}",
            f"{metrics_with_nan['answer_relevancy']:.3f}",
            f"{metrics_with_nan['context_precision']:.3f}",
            f"{metrics_with_nan['context_recall']:.3f}",
            f"{metrics_with_nan['answer_correctness']:.3f}",
            ""
        ])
        writer.writerow([
            timestamp,
            "--- MEDIA CLEAN (solo no-error, NaN=0) ---",
            f"{metrics_clean['faithfulness']:.3f}",
            f"{metrics_clean['answer_relevancy']:.3f}",
            f"{metrics_clean['context_precision']:.3f}",
            f"{metrics_clean['context_recall']:.3f}",
            f"{metrics_clean['answer_correctness']:.3f}",
            ""
        ])

    print(f"Risultati salvati in: {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    llm_model_name, embedding_model_name, search_technique, chunking, version = args.llm_model, args.embedding_model, args.search, args.chunking, args.version
    use_reranking = args.reranking
    rerank_method = args.rerank_method

    embedding_model, vectorstores, llm, COLLECTION_NAMES, qdrant_client, reranker = init_components(embedding_model_name=embedding_model_name, llm_model_name=llm_model_name)
    corpus, corpus_texts = build_corpus(qdrant_client, COLLECTION_NAMES)
    spacy_tokenizer = build_spacy_tokenizer()
    bm25 = build_bm25(corpus_texts, spacy_tokenizer)
    
    # Crea il grafo RAG
    rag_graph = build_rag_graph()
    
    if search_technique == "dense":
        evaluate_variant(rag_graph, embedding_model, vectorstores, corpus, bm25, spacy_tokenizer, reranker, llm, embedding_model_name, llm_model_name, chunking, "dense", version, use_reranking, rerank_method)
    elif search_technique == "sparse":
        evaluate_variant(rag_graph, embedding_model, vectorstores, corpus, bm25, spacy_tokenizer, reranker, llm, embedding_model_name, llm_model_name, chunking, "sparse", version, use_reranking, rerank_method)
    elif search_technique == "hybrid":
        evaluate_variant(rag_graph, embedding_model, vectorstores, corpus, bm25, spacy_tokenizer, reranker, llm, embedding_model_name, llm_model_name, chunking, "hybrid", version, use_reranking, rerank_method)
    else:
        evaluate_variant(rag_graph, embedding_model, vectorstores, corpus, bm25, spacy_tokenizer, reranker, llm, embedding_model_name, llm_model_name, chunking, "dense", version, use_reranking, rerank_method)
        evaluate_variant(rag_graph, embedding_model, vectorstores, corpus, bm25, spacy_tokenizer, reranker, llm, embedding_model_name, llm_model_name, chunking, "sparse", version, use_reranking, rerank_method)
        evaluate_variant(rag_graph, embedding_model, vectorstores, corpus, bm25, spacy_tokenizer, reranker, llm, embedding_model_name, llm_model_name, chunking, "hybrid", version, use_reranking, rerank_method)