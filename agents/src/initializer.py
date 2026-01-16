import os, json
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_ollama import OllamaLLM

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_google_vertexai import ChatVertexAI
from google.oauth2 import service_account


def load_google_creds():
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
        creds_dict = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
        creds = service_account.Credentials.from_service_account_info(creds_dict)
    else:
        creds = None
    return creds


def load_env_config(): 
    OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
    QDRANT_URL = os.environ["QDRANT_URL"]
    COLLECTION_NAMES = os.getenv("COLLECTION_NAMES", "").split(",")
    COLLECTION_NAMES = [c.strip() for c in COLLECTION_NAMES if c.strip()]
    return OLLAMA_BASE_URL, QDRANT_URL, COLLECTION_NAMES


def build_embeddings(ollama_base_url, model_name="nomic"): 
    if model_name == "nomic":
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=ollama_base_url
        )
    elif model_name == "e5":
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-base-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
    elif model_name == "all-mpnet":
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            encode_kwargs={"normalize_embeddings": True}
        )

    return embeddings


def build_qdrant_client(qdrant_url): 
    qdrant_client = QdrantClient(url=qdrant_url)
    return qdrant_client


def build_vectorstores(qdrant_client, embeddings, qdrant_url, collection_names):
    vectorstores = {}
    try:
        existing = qdrant_client.get_collections().collections
        existing_names = [c.name for c in existing]
    except Exception as e:
        print(f"[WARN] Impossibile ottenere elenco collezioni: {e}")
        existing_names = []

    for name in collection_names:
        if name not in existing_names:
            print(f"[WARN] Collezione inesistente su Qdrant: {name} (saltata)")
            continue
        try:
            store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                collection_name=name,
                url=qdrant_url,
            )
            vectorstores[name] = store
        except Exception as e:
            print(f"[WARN] Errore su {name}: {e}")

    if not vectorstores:
        raise RuntimeError("Nessuna collezione valida trovata in Qdrant. Controlla i nomi in .env!")
    
    return vectorstores


def test_connection(vectorstores, embeddings):
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


def build_llm(model_name: str, ollama_base_url: str, creds):
    if model_name == "llama-local":
        llm = OllamaLLM(model="llama3.2:3b", base_url=ollama_base_url)

    elif model_name == "gemini":
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
        )

    elif model_name == "llama-api":
        llm = ChatVertexAI(
            model="llama-3.3-70b-instruct-maas",
            location="us-central1",
            temperature=0,
            max_output_tokens=1024,
            credentials=creds,
        )

    return llm


def init_components(embedding_model_name: str, llm_model_name: str):
    OLLAMA_BASE_URL, QDRANT_URL, COLLECTION_NAMES = load_env_config()
    creds = load_google_creds()
    embeddings = build_embeddings(OLLAMA_BASE_URL, model_name=embedding_model_name)
    qdrant_client = build_qdrant_client(QDRANT_URL)
    vectorstores = build_vectorstores(qdrant_client, embeddings, QDRANT_URL, COLLECTION_NAMES)
    llm = build_llm(llm_model_name, OLLAMA_BASE_URL, creds)

    return embeddings, vectorstores, llm, COLLECTION_NAMES, qdrant_client