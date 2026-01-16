import json
import os
import csv
from datetime import datetime, timezone
from pathlib import Path
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.metrics._answer_relevance import answer_relevancy
from ragas import evaluate
from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI

from query_processing import answer_query_dense, answer_query_bm25, answer_query_hybrid
from initializer import init_components


def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def evaluate_variant(name: str, answer_func, version: str):
    print(f"\n=== Valutazione variante: {name.upper()} ===")
    base_dir = Path("evaluations") / version
    base_dir.mkdir(parents=True, exist_ok=True)
    #csv_path = base_dir / f"ragas_results_{name.replace(' ', '_')}.csv"
    #csv_path = base_dir / f"llama-70b-ragas-results-{name.replace(' ', '_')}.csv"
    csv_path = base_dir / f"gemini-all-mpnet-base-v2-ragas-results-{name.replace(' ', '_')}.csv"

    VALIDATION_DIR = Path(__file__).resolve().parent / "validation_set"
    validation_data = []
    for json_file in sorted(VALIDATION_DIR.glob("*.json")):
        print(f"  → Trovato file: {json_file.name}")
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
    ground_truths = [d.get("ground_truth", None) for d in validation_data]

    answers, retrieved_contexts = [], []

    print(f"\nGenerazione risposte con variante {name} ({len(questions)} domande)...")
    for i, q in enumerate(questions, start=1):
        print(f" → [{i}/{len(questions)}] {q}")
        try:
            response, ctxs = answer_func(q)
            answers.append(response.split(":", 1)[-1].strip())
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
    llm = get_llm()
    metrics_list = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]

    embeddings, vectorstores, llm, COLLECTION_NAMES, qdrant_client = init_components(embedding_model_name="nomic", llm_model_name="gemini")
    result = evaluate(dataset=dataset, metrics=metrics_list, llm=llm, embeddings=embeddings)
    result_df = result.to_pandas()
    results = result_df.mean(numeric_only=True).to_dict()

    print("\nRISULTATI RAGAS GLOBALI:")
    for k, v in results.items():
        print(f" - {k}: {v:.3f}")

    save_results_to_csv(csv_path, dataset, result_df, results)
    print(f"Risultati salvati in {csv_path}")

    #invio automatico a LangSmith
    langsmith_key = os.getenv("LANGCHAIN_API_KEY")
    if langsmith_key:
        try:
            client = Client()
            client.create_run(
                name=f"RAG Evaluation {name.upper()} - UNIPG ({version})",
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

    print(f"=== Fine valutazione {name.upper()} ===\n")


def save_results_to_csv(csv_path: Path, dataset: Dataset, result_df, metrics):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "question", "answer", "faithfulness",
                "answer_relevancy", "context_precision", "context_recall",
                "answer_correctness"
            ])

        for i in range(len(dataset)):
            row = result_df.iloc[i]
            writer.writerow([
                timestamp,
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
            "=== MEDIA TOTALE ===",
            "",
            f"{metrics['faithfulness']:.3f}",
            f"{metrics['answer_relevancy']:.3f}",
            f"{metrics['context_precision']:.3f}",
            f"{metrics['context_recall']:.3f}",
            f"{metrics['answer_correctness']:.3f}",
        ])

    print(f"Risultati individuali e medi salvati in: {csv_path}")

if __name__ == "__main__":
    version = os.getenv("RAG_EVAL_VERSION", "semantic chunking")

    evaluate_variant("dense-semantic chunking", answer_query_dense, version)
    evaluate_variant("sparse-semantic chunking", answer_query_bm25, version)
    evaluate_variant("hybrid-semantic chunking", answer_query_hybrid, version)
