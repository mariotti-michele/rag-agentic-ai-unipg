import json
import os
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

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
from query_processing import answer_query_dense, answer_query_bm25, answer_query_hybrid, classify_query

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Valutazione RAG automatica con RAGAS")
    parser.add_argument("--llm-model", type=str, default="vllm",
                        choices=["llama-local", "gemini", "llama-api", "vllm"],
                        help="Seleziona il modello da usare")
    parser.add_argument("--embedding-model", type=str, default="nomic",
                        choices=["nomic", "e5", "all-mpnet", "bge"],
                        help="Seleziona il modello di embedding da usare")
    parser.add_argument("--search", type=str, default="all",
                        choices=["dense", "sparse", "hybrid", "all"],
                        help="Seleziona tecnica di ricerca da utilizzare (default: all)")
    parser.add_argument("--chunking", type=str, default="section",
                        choices=["fixed", "document-structure", "section", "semantic"],
                        help="Tipo di chunking usato per creare la collezione (default: section)")
    parser.add_argument("--version", type=str, default="v0",
                        help="Versione del modello valutato (default: v0)")
    parser.add_argument("--reranking", action="store_true",
                        help="Attiva il re-ranking dei documenti")
    parser.add_argument("--rerank-method", type=str, default="cross_encoder",
                        choices=["cross_encoder", "llm"],
                        help="Metodo di re-ranking: cross_encoder (veloce) o llm (accurato)")
    args = parser.parse_args()
    return args


def evaluate_variant(answer_func, llm, embedding_model, llm_model_name: str, embedding_model_name: str, chunking: str, search: str, version: str):
    print(f"\nValutazione variante: \nllm model: {llm_model_name} \nembedding model: {embedding_model_name} \nchunking: {chunking} \nsearch: {search} \nversion: {version}")

    base_dir = Path("evaluations_results") / llm_model_name / embedding_model_name / chunking / (search + f"_{version}")
    base_dir.mkdir(parents=True, exist_ok=True)
    name = f"{llm_model_name}-{embedding_model_name}-{chunking}-{search}-{version}"
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

    answers, retrieved_contexts = [], []

    print(f"\nGenerazione risposte con variante {name} ({len(questions)} domande)...")
    for i, q in enumerate(questions, start=1):
        print(f" -> [{i}/{len(questions)}] {q}")
        try:
            response, ctxs = answer_func(q)
            answers.append(response)
            retrieved_contexts.append(ctxs)
        except Exception as e:
            print(f"Errore durante la domanda '{q}': {e}")
            answers.append("")
            retrieved_contexts.append([])

    dataset = Dataset.from_dict({
        "question": questions,
        "contexts": retrieved_contexts,
        "answer": answers,
        "ground_truth": ground_truths
    })

    print(f"\nValutazione Ragas per {name}...")
    metrics_list = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]

    run_config = RunConfig(
        timeout=300,  # timeout in secondi, prima
        max_workers=2,  # workers paralleli
        max_wait=360  # tempo massimo di attesa
    )

    result = evaluate(
        dataset=dataset, 
        metrics=metrics_list, 
        llm=llm, 
        embeddings=embedding_model,
        run_config=run_config
    )
    result_df = result.to_pandas()
    results = result_df.mean(numeric_only=True).to_dict()

    print("\nRISULTATI RAGAS MEDIE GLOBALI:")
    for k, v in results.items():
        print(f" - {k}: {v:.3f}")

    save_results_to_csv(csv_path, dataset, result_df, results, search_technique=search)
    print(f"Risultati salvati in {csv_path}")

    langsmith_key = os.getenv("LANGCHAIN_API_KEY")
    if langsmith_key and False: # Disabilitato temporaneamente
        try:
            client = Client()
            client.create_run(
                name=f"Evaluation RAG UNIPG {name}",
                run_type="chain",
                inputs={"questions": questions},
                outputs={"results": dict(results)},
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
            print(f"[WARN] Errore durante l’invio a LangSmith: {e}")
    else:
        print("Nessuna API key LangSmith trovata — risultati solo in locale.")


def save_results_to_csv(csv_path: Path, dataset: Dataset, result_df, metrics, search_technique: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Ricerca " + search_technique])
            writer.writerow([
                "question", "answer", "faithfulness",
                "answer_relevancy", "context_precision", "context_recall",
                "answer_correctness"
            ])

        for i in range(len(dataset)):
            row = result_df.iloc[i]
            writer.writerow([
                dataset[i]["question"],
                dataset[i]["answer"],
                f"{row['faithfulness']:.3f}",
                f"{row['answer_relevancy']:.3f}",
                f"{row['context_precision']:.3f}",
                f"{row['context_recall']:.3f}",
                f"{row['answer_correctness']:.3f}",
            ])

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            "--- MEDIA TOTALE ---",
            f"{metrics['faithfulness']:.3f}",
            f"{metrics['answer_relevancy']:.3f}",
            f"{metrics['context_precision']:.3f}",
            f"{metrics['context_recall']:.3f}",
            f"{metrics['answer_correctness']:.3f}",
        ])

    print(f"Risultati salvati in: {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    llm_model_name, embedding_model_name, search_technique, chunking, version = args.llm_model, args.embedding_model, args.search, args.chunking, args.version
    use_reranking = args.reranking
    rerank_method = args.rerank_method

    embedding_model, vectorstores, llm, COLLECTION_NAMES, qdrant_client = init_components(embedding_model_name=embedding_model_name, llm_model_name=llm_model_name)
    corpus, corpus_texts = build_corpus(qdrant_client, COLLECTION_NAMES)
    spacy_tokenizer = build_spacy_tokenizer()
    bm25 = build_bm25(corpus_texts, spacy_tokenizer)
    
    dense_func = lambda q: answer_query_dense(q, embedding_model, embedding_model_name, vectorstores, llm, classify_query(llm, q), use_reranking, rerank_method)
    sparse_func = lambda q: answer_query_bm25(q, corpus, bm25, spacy_tokenizer, llm, classify_query(llm, q))
    hybrid_func = lambda q: answer_query_hybrid(q, embedding_model, embedding_model_name, vectorstores, corpus, bm25, spacy_tokenizer, llm, classify_query(llm, q), use_reranking, rerank_method)
    
    if search_technique == "dense":
        evaluate_variant(dense_func, llm, embedding_model, llm_model_name, embedding_model_name, chunking, "dense", version)
    elif search_technique == "sparse":
        evaluate_variant(sparse_func, llm, embedding_model, llm_model_name, embedding_model_name, chunking, "sparse", version)
    elif search_technique == "hybrid":
        evaluate_variant(hybrid_func, llm, embedding_model, llm_model_name, embedding_model_name, chunking, "hybrid", version)
    else:
        evaluate_variant(dense_func, llm, embedding_model, llm_model_name, embedding_model_name, chunking, "dense", version)
        evaluate_variant(sparse_func, llm, embedding_model, llm_model_name, embedding_model_name, chunking, "sparse", version)
        evaluate_variant(hybrid_func, llm, embedding_model, llm_model_name, embedding_model_name, chunking, "hybrid", version)