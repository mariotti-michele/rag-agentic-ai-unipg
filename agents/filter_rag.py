import os, json
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate

#llama locale:
from langchain_ollama import OllamaLLM

#gemini:
from langchain_google_genai import ChatGoogleGenerativeAI

#llama 3.3 70b api:
from langchain_google_vertexai import ChatVertexAI
from google.oauth2 import service_account

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
# BM25
from rank_bm25 import BM25Okapi



if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
    creds_dict = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    creds = service_account.Credentials.from_service_account_info(creds_dict)
else:
    creds = None


OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
QDRANT_URL = os.environ["QDRANT_URL"]
COLLECTION_NAMES = os.getenv("COLLECTION_NAMES", "").split(",")
COLLECTION_NAMES = [c.strip() for c in COLLECTION_NAMES if c.strip()]


rag_prompt_template = """Sei un assistente accademico.
Hai accesso a estratti di documenti ufficiali.

Usa SOLO il contesto fornito per rispondere.
Se il contesto contiene il termine o l'argomento richiesto, indica che è presente e copia il testo più rilevante.
Non aggiungere nulla di tuo e non inventare.

Se davvero non ci sono riferimenti nemmeno parziali, rispondi esattamente: "Non presente nei documenti".

Domanda: {question}

Contesto:
{context}

Risposta:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=rag_prompt_template,
)

classifier_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Sei un classificatore di query.

Se la domanda riguarda saluti, domande generiche, curiosità non legate a regolamenti o procedure accademiche → rispondi SOLO con: semplice

Se la domanda richiede informazioni ufficiali su corsi, tesi, tirocini, lauree, esami, regolamenti, scadenze → rispondi SOLO con: rag

Domanda: {question}
Categoria:""",
)

simple_prompt_template = """Sei un assistente accademico gentile.
Rispondi in modo breve e diretto alla domanda generica seguente:

Domanda: {question}
Risposta:"""


embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
)

#print("Connettendo a Qdrant...")
qdrant_client = QdrantClient(url=QDRANT_URL)


vectorstores = {}
#print("Verifica collezioni in Qdrant...")

try:
    existing = qdrant_client.get_collections().collections
    existing_names = [c.name for c in existing]
    #print(f"Trovate {len(existing_names)} collezioni in Qdrant.")
except Exception as e:
    print(f"[WARN] Impossibile ottenere elenco collezioni: {e}")
    existing_names = []

for name in COLLECTION_NAMES:
    if name not in existing_names:
        #print(f"[WARN] Collezione inesistente su Qdrant: {name} (saltata)")
        continue
    try:
        store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=name,
            url=QDRANT_URL,
        )
        vectorstores[name] = store
        #print(f"Collezione connessa: {name}")
    except Exception as e:
        print(f"[WARN] Errore su {name}: {e}")

if not vectorstores:
    raise RuntimeError("Nessuna collezione valida trovata in Qdrant. Controlla i nomi in .env!")

print(f"Connesso a {len(vectorstores)} collezioni valide:")
for name in vectorstores:
    print(f"  - {name}")


#print("Inizializzando LLM...")

import argparse

parser = argparse.ArgumentParser(description="Sistema Q&A con modelli selezionabili")
parser.add_argument("--model", type=str, default="gemini",
                    choices=["llama-local", "gemini", "llama-api"],
                    help="Seleziona il modello da usare")
args = parser.parse_args()

print(f"Inizializzando LLM con modello: {args.model}")

if args.model == "llama-local":
    # llama locale
    llm = OllamaLLM(model="llama3.2:3b", base_url=OLLAMA_BASE_URL)

elif args.model == "gemini":
    # gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )

elif args.model == "llama-api":
    # llama 3.3 70b API (Vertex AI)
    llm = ChatVertexAI(
        model="llama-3.3-70b-instruct-maas",
        location="us-central1",
        temperature=0,
        max_output_tokens=1024,
        credentials=creds,
    )


all_texts = []

try:
    existing = qdrant_client.get_collections().collections
    existing_names = [c.name for c in existing]
except Exception as e:
    print(f"[WARN] Impossibile leggere elenco collezioni da Qdrant: {e}")
    existing_names = []

for COLLECTION_NAME in COLLECTION_NAMES:
    if COLLECTION_NAME not in existing_names:
        #print(f"[WARN] Collezione inesistente su Qdrant: {COLLECTION_NAME} (saltata)")
        continue

    #print(f"Leggendo documenti da collezione: {COLLECTION_NAME}")
    scroll_filter = None
    total_texts = 0

    while True:
        try:
            points, next_page = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                with_payload=True,
                limit=1000,
                offset=scroll_filter
            )
        except Exception as e:
            print(f"[WARN] Errore nel leggere {COLLECTION_NAME}: {e}")
            break

        if not points:
            break

        for p in points:
            text = p.payload.get("page_content", "")
            if text:
                all_texts.append(text)
                total_texts += 1

        if next_page is None:
            break
        scroll_filter = next_page

    if total_texts == 0:
        print(f"[WARN] Nessun testo trovato in {COLLECTION_NAME}")
    #else:
        #print(f"{total_texts} testi caricati da {COLLECTION_NAME}")

#print(f"\nTotale complessivo di testi caricati: {len(all_texts)}")

if not all_texts:
    raise RuntimeError(
        "Nessun testo trovato in nessuna collezione! "
        "Controlla che i nomi in COLLECTION_NAMES coincidano con quelli effettivi su Qdrant."
    )


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_texts)

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)

tokenized_corpus = [word_tokenize(text.lower()) for text in all_texts]
bm25 = BM25Okapi(tokenized_corpus)


def tfidf_search(query: str, k: int = 5):
    query_vec = vectorizer.transform([query])
    scores = (tfidf_matrix @ query_vec.T).toarray().ravel()
    top_indices = np.argsort(scores)[::-1][:k]
    results = [(all_texts[i], scores[i]) for i in top_indices]
    return results

def bm25_search(query: str, k: int = 5):
    """Ricerca sparse basata su BM25"""
    query_tokens = word_tokenize(query.lower())
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    results = [(all_texts[i], scores[i]) for i in top_indices]
    return results


def classify_query(query: str) -> str:
    """Classifica la query in 'semplice' o 'rag'"""
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
        print(f"Errore classificazione: {e}")
        return "rag"


def answer_query_dense(query: str):
    """Usa tutte le collezioni Dense (Qdrant) e ritorna risposta + contesto"""
    vec = embeddings.embed_query(query)
    all_docs = []

    for name, store in vectorstores.items():
        try:
            docs = store.similarity_search_by_vector(vec, k=5)
            for d in docs:
                d.metadata["collection"] = name
            all_docs.extend(docs)
        except Exception as e:
            print(f"[WARN] Errore su {name}: {e}")

    seen = set()
    unique_docs = []
    for d in all_docs:
        text = d.page_content.strip()
        if text not in seen:
            seen.add(text)
            unique_docs.append(d)
        if len(unique_docs) == 5:
            break

    if not unique_docs:
        return "Non presente nei documenti.", []

    context = "\n\n".join(
        [f"[Fonte {i+1}] ({doc.metadata.get('collection', 'N/A')})\n{doc.page_content}"
         for i, doc in enumerate(unique_docs)]
    )

    prompt = f"""{QA_CHAIN_PROMPT.format(context=context, question=query)}

Rispondi in un unico paragrafo chiaro e completo, senza aggiungere sezioni o titoli.
"""
    answer = llm.invoke(prompt)
    if hasattr(answer, "content"):
        answer = answer.content

    return f"Risposta nomic-embed-text: {answer}", [doc.page_content for doc in unique_docs]



def answer_query_tfidf(query: str):
    """Usa solo Sparse (TF-IDF)"""
    tfidf_results = tfidf_search(query, k=5)
    if not tfidf_results:
        return "Non presente nei documenti"

    context = "\n\n".join(
        [f"[Fonte {i+1}] (TF-IDF, score={score:.3f})\n{text}"
         for i, (text, score) in enumerate(tfidf_results)]
    )

    prompt = f"""{QA_CHAIN_PROMPT.format(context=context, question=query)}

Rispondi in un unico paragrafo chiaro e completo, senza aggiungere sezioni o titoli.
"""
    answer = llm.invoke(prompt)
    if hasattr(answer, "content"):
        answer = answer.content

    return f"Risposta TF-IDF: {answer}"



def answer_query_bm25(query: str):
    """Usa BM25 e ritorna risposta + contesto"""
    bm25_results = bm25_search(query, k=5)
    if not bm25_results:
        return "Non presente nei documenti.", []

    context_texts = [text for text, score in bm25_results]

    context = "\n\n".join(
        [f"[Fonte {i+1}] (BM25, score={score:.3f})\n{text}"
         for i, (text, score) in enumerate(bm25_results)]
    )

    prompt = f"""{QA_CHAIN_PROMPT.format(context=context, question=query)}

Rispondi in un unico paragrafo chiaro e completo, senza aggiungere sezioni o titoli.
"""
    answer = llm.invoke(prompt)
    if hasattr(answer, "content"):
        answer = answer.content

    return f"Risposta BM25: {answer}", context_texts




def compare_dense_vs_tfidf(query: str):
    """Confronta Dense vs Sparse"""
    print(f"\n Query: {query}\n{'='*70}")

    dense_answer = answer_query_dense(query)
    print("\n--- Risposta Dense (Qdrant) ---")
    print(dense_answer)

    sparse_answer = answer_query_tfidf(query)
    print("\n--- Risposta Sparse (TF-IDF) ---")
    print(sparse_answer)

    print("="*70)


def hybrid_search(query: str, alpha: float = 0.5, k: int = 5):
    """Retrieval ibrido combinato su tutte le collezioni. Ritorna risposta + contesto"""
    dense_vec = embeddings.embed_query(query)
    dense_results = {}

    # Recupera documenti densi
    for name, store in vectorstores.items():
        try:
            docs = store.similarity_search_with_score_by_vector(dense_vec, k=5)
            for doc, score in docs:
                dense_results[doc.page_content] = 1 - score
        except Exception as e:
            print(f"[WARN] Errore su {name}: {e}")

    # Recupera documenti sparsi
    tfidf_results = bm25_search(query, k=10)
    sparse_results = {text: score for text, score in tfidf_results}

    # Fusione punteggi
    all_texts = list(set(dense_results.keys()) | set(sparse_results.keys()))
    dense_scores = np.array([dense_results.get(t, 0) for t in all_texts])
    sparse_scores = np.array([sparse_results.get(t, 0) for t in all_texts])

    if dense_scores.max() > 0:
        dense_scores /= dense_scores.max()
    if sparse_scores.max() > 0:
        sparse_scores /= sparse_scores.max()

    hybrid_scores = alpha * dense_scores + (1 - alpha) * sparse_scores
    sorted_indices = np.argsort(hybrid_scores)[::-1][:k]
    top_texts = [all_texts[i] for i in sorted_indices]
    top_scores = hybrid_scores[sorted_indices]

    context = "\n\n".join(
        [f"[Fonte {i+1}] (Hybrid score={s:.3f})\n{t}" for i, (t, s) in enumerate(zip(top_texts, top_scores))]
    )

    prompt = f"""{QA_CHAIN_PROMPT.format(context=context, question=query)}

Rispondi in un unico paragrafo chiaro e completo, senza aggiungere sezioni o titoli.
"""
    answer = llm.invoke(prompt)
    if hasattr(answer, "content"):
        answer = answer.content

    return f"Risposta ibrida (α={alpha}): {answer}", top_texts

    

# def answer_query(query: str):
#     try:
#         # STEP 1: classifica la query
#         mode = classify_query(query)
#         print(f"Classificazione query: {mode}")

#         # STEP 2: risposta semplice
#         if mode == "semplice":
#             prompt = simple_prompt_template.format(question=query)
#             answer = llm.invoke(prompt)
#             if hasattr(answer, "content"):
#                 answer = answer.content
#             return f"Risposta semplice: {answer}"

#         # STEP 3: risposta RAG
#         print("Processando query con RAG...")
#         vec = embeddings.embed_query(query)
#         docs = vectorstore.similarity_search_by_vector(vec, k=8)

#         seen = set()
#         unique_docs = []
#         for d in docs:
#             text = d.page_content.strip()
#             if text not in seen:
#                 seen.add(text)
#                 unique_docs.append(d)
#             if len(unique_docs) == 5:
#                 break

#         if not unique_docs:
#             return "Non presente nei documenti"

#         context = "\n\n".join(
#             [f"[Fonte {i+1}] ({doc.metadata.get('source_url', 'N/A')})\n{doc.page_content}"
#              for i, doc in enumerate(unique_docs)]
#         )

#         prompt = f"""{QA_CHAIN_PROMPT.format(context=context, question=query)}

#     Rispondi in un unico paragrafo chiaro e completo, senza aggiungere sezioni o titoli.
#     """
#         answer = llm.invoke(prompt)
#         if hasattr(answer, "content"):
#             answer = answer.content

#         main_source = unique_docs[0].metadata.get("source_url", "N/A")
#         response = f"Risposta: {answer}\n"
#         response += f"\nPer ulteriori informazioni consulta il seguente link: {main_source}\n"

#         if unique_docs:
#             response += f"\nFonti consultate ({len(unique_docs)} documenti):"
#             for i, doc in enumerate(unique_docs, 1):
#                 preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
#                 response += f"\n{i}. {preview}"
#         return response

#     except Exception as e:
#         return f"Errore durante la query: {e}"


def test_connection():
    """Testa la connessione con tutte le collezioni valide"""
    if not vectorstores:
        print("Nessuna collezione valida trovata.")
        return False

    vec = embeddings.embed_query("test")
    ok = False

    for name, store in vectorstores.items():
        try:
            docs = store.max_marginal_relevance_search_by_vector(
                vec, k=3, fetch_k=10, lambda_mult=0.5
            )
            print(f"[OK] Connessione riuscita per {name} ({len(docs)} risultati)")
            ok = True
        except Exception as e:
            print(f"[WARN] Test fallito su {name}: {e}")

    return ok


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
                    # confronto automatico Dense vs Sparse
                    compare_dense_vs_tfidf(q)
                    print("\n--- 📙 Risposta Ibrida (Dense + TF-IDF) ---")
                    hybrid_answer = hybrid_search(q, alpha=0.6, k=5)
                    print(hybrid_answer)

                print("-" * 50)
            else:
                print("Inserisci una domanda valida.")
        except KeyboardInterrupt:
            print("\nUscita...")
            break
        except Exception as e:
            print(f"Errore: {e}")



    # while True:
    #     try:
    #         q = input("Domanda: ")
    #         if q.lower() in ["exit", "quit"]:
    #             break
    #         if q.strip():
    #             result = answer_query(q)
    #             print(f"\n{result}\n")
    #             print("-" * 50)
    #         else:
    #             print("Inserisci una domanda valida.")
    #     except KeyboardInterrupt:
    #         print("\nUscita...")
    #         break
    #     except Exception as e:
    #         print(f"Errore: {e}")