import os, json
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

#llama locale:
from langchain_ollama import OllamaLLM

#gemini:
from langchain_google_genai import ChatGoogleGenerativeAI

#llama 3.3 70b api:
from langchain_google_vertexai import ChatVertexAI
from google.oauth2 import service_account

import numpy as np
# TF-IDF
# from sklearn.feature_extraction.text import TfidfVectorizer

# BM25
from rank_bm25 import BM25Okapi

import spacy


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
Sei un classificatore di query accademiche.

Devi scegliere una sola categoria tra:
- "semplice"
- "rag"

Rispondi SOLO con una di queste due parole.

Regole:
- Se la domanda contiene solo saluti, convenevoli o curiosità non universitarie (es. "ciao", "buongiorno", "come stai", "grazie", "che tempo fa", "chi sei") → rispondi: semplice
- In TUTTI gli altri casi, anche se la domanda è breve ma riguarda università, corsi, lezioni, orari, esami, tesi, lauree, tirocini, regolamenti, o informazioni accademiche → rispondi: rag

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

# embeddings = HuggingFaceEmbeddings(
#     model_name="intfloat/e5-base-v2",
#     encode_kwargs={"normalize_embeddings": True}
# )

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     encode_kwargs={"normalize_embeddings": True}
# )

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


corpus = []

try:
    existing = qdrant_client.get_collections().collections
    existing_names = [c.name for c in existing]
except Exception as e:
    print(f"[WARN] Impossibile leggere elenco collezioni da Qdrant: {e}")
    existing_names = []

for COLLECTION_NAME in COLLECTION_NAMES:
    if COLLECTION_NAME not in existing_names:
        continue

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
                corpus.append({
                    "text": text,
                    "collection": COLLECTION_NAME,
                    "source_url": p.payload.get("source_url", ""),
                    "section_path": p.payload.get("section_path", ""),
                    "doc_id": p.payload.get("doc_id", ""),
                })
                total_texts += 1

        if next_page is None:
            break
        scroll_filter = next_page

    if total_texts == 0:
        print(f"[WARN] Nessun testo trovato in {COLLECTION_NAME}")

if not corpus:
    raise RuntimeError("Nessun testo trovato in nessuna collezione!")

print(f"[INFO] Corpus completo: {len(corpus)} chunk con metadata")


all_texts = [c["text"] for c in corpus]

# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(all_texts)

nlp = spacy.load("it_core_news_sm", disable=["parser", "ner", "tagger", "lemmatizer"])

def spacy_tokenize(text: str):
    doc = nlp(text.lower())
    return [t.text for t in doc if not t.is_space and not t.is_punct and not t.is_stop]

# nlp = spacy.load("it_core_news_sm", disable=["parser", "ner"])
# def spacy_tokenize(text: str):
#     doc = nlp(text.lower())
#     return [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_space]


tokenized_corpus = [spacy_tokenize(t) for t in all_texts]
bm25 = BM25Okapi(tokenized_corpus)


# def tfidf_search_idx(query: str, k: int = 5):
#     qv = vectorizer.transform([query])
#     scores = (tfidf_matrix @ qv.T).toarray().ravel()
#     top_idx = np.argsort(scores)[::-1][:k]
#     return [(i, float(scores[i])) for i in top_idx]

def bm25_search_idx(query: str, k: int = 5):
    qtoks = spacy_tokenize(query)
    scores = bm25.get_scores(qtoks)
    top_idx = np.argsort(scores)[::-1][:k]
    return [(i, float(scores[i])) for i in top_idx]


def dense_search(query: str, top_k: int = 5):
    vec = embeddings.embed_query(query)
    hits = []

    for name, store in vectorstores.items():
        try:
            docs_scores = store.similarity_search_with_score_by_vector(vec, k=top_k)

            print("\n[DEBUG dense scores]")
            for doc, score in docs_scores:
                print(f"score={score:.6f} | {doc.page_content[:80]}")
            for doc, score in docs_scores:
                hits.append({
                    "text": doc.page_content,
                    "collection": name,
                    "source_url": doc.metadata.get("source_url", ""),
                    "section_path": doc.metadata.get("section_path", ""),
                    "doc_id": doc.metadata.get("doc_id", ""),
                    "score": float(score),
                })
        except Exception as e:
            print(f"[WARN] Errore su {name}: {e}")

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:top_k]


def reciprocal_rank_fusion_docs(dense_docs, sparse_docs, alpha=60, k=5):
    combined = {}
    for rank, d in enumerate(dense_docs):
        docid = d.get("doc_id") if d.get("doc_id") else d["text"]
        combined[docid] = combined.get(docid, 0) + 1 / (alpha + rank + 1)
    for rank, d in enumerate(sparse_docs):
        docid = d.get("doc_id") if d.get("doc_id") else d["text"]
        combined[docid] = combined.get(docid, 0) + 1 / (alpha + rank + 1)

    ranked_ids = [rid for rid, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]]
    merged = {(d.get("doc_id") if d.get("doc_id") else d["text"]): d for d in dense_docs + sparse_docs}
    return [merged[rid] for rid in ranked_ids if rid in merged]


def hybrid_search(query: str, alpha: int = 60, k: int = 5):
    dense_docs = dense_search(query)
    sparse_idxs = bm25_search_idx(query, k=10)
    sparse_docs = [{
        **corpus[i],
        "score": score,
    } for i, score in sparse_idxs]

    if not dense_docs and not sparse_docs:
        return "Non presente nei documenti", []

    merged_docs = reciprocal_rank_fusion_docs(dense_docs, sparse_docs, alpha=alpha, k=k)

    context = ""
    for i, d in enumerate(merged_docs, 1):
        section = f" | Sezione: {d.get('section_path','')}" if d.get("section_path") else ""
        context += f"[Fonte {i}] ({d.get('collection','N/A')}){section}\n{d.get('text','')}\n\n"

    prompt = f"""{QA_CHAIN_PROMPT.format(context=context, question=query)}

Rispondi in un unico paragrafo chiaro e completo, senza aggiungere sezioni o titoli.
"""
    answer = llm.invoke(prompt)
    if hasattr(answer, "content"):
        answer = answer.content

    return f"Risposta ibrida (α={alpha}): {answer}", [d["text"] for d in merged_docs]


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



def test_connection():
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
                    print("\n--- Risposta Ibrida (Dense + BM25) ---")
                    hybrid_answer = hybrid_search(q)
                    print(hybrid_answer)

                print("-" * 50)
            else:
                print("Inserisci una domanda valida.")
        except KeyboardInterrupt:
            print("\nUscita...")
            break
        except Exception as e:
            print(f"Errore: {e}")