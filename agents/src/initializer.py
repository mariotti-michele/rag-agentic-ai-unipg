import os, json
import requests
import time
import urllib3
import warnings

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from langchain_ollama import OllamaLLM

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_google_vertexai import ChatVertexAI
from google.oauth2 import service_account

from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from typing import Any, List, Optional

# Disabilita warning SSL
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class VLLMChat(BaseLLM):
    api_url: str
    api_key: str
    model: str = "RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8"
    temperature: float = 0.2

    def _endpoint(self) -> str:
        u = self.api_url.strip()

        if u.rstrip("/").endswith("/llm/chat/completions"):
            return u.rstrip("/")
        if u.rstrip("/").endswith("/llm"):
            return u.rstrip("/") + "/chat/completions"
        return u.rstrip("/") + "/llm/chat/completions"

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        generations = []
        endpoint = self._endpoint()

        for prompt in prompts:
            response = requests.post(
                endpoint,
                headers={
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                },
                verify=False,
                timeout=60,
            )
            
            if response.status_code >= 400:
                print(f"[ERROR {response.status_code}] Response text: {response.text}")
                print(f"[ERROR {response.status_code}] Prompt length: {len(prompt)}")
            
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            generations.append([Generation(text=content)])

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "vllm-chat"


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
    VLLM_API_URL = os.getenv("VLLM_API_URL", "")
    VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
    BGE_API_URL = os.getenv("BGE_EMBED_MODEL_API_URL", "")
    BGE_API_KEY = os.getenv("BGE_EMBED_MODEL_API_KEY", "")
    return OLLAMA_BASE_URL, QDRANT_URL, COLLECTION_NAMES, VLLM_API_URL, VLLM_API_KEY, BGE_API_URL, BGE_API_KEY


class BGEEmbeddings(Embeddings):
    api_url: str
    api_key: str
    model: str = "BAAI/bge-m3"
    max_retries: int = 5
    initial_retry_delay: int = 3

    def __init__(self, api_url: str, api_key: str, model: str = "BAAI/bge-m3"):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def _endpoint(self) -> str:
        u = self.api_url.strip().rstrip("/")
        if u.endswith("/embeddings/embed"):
            return u
        if u.endswith("/embeddings"):
            return u + "/embed"
        if u.endswith("/llm"):
            return u.rstrip("/llm") + "/embeddings/embed"
        return u + "/embeddings/embed"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 1
        all_embeddings = []
        endpoint = self._endpoint()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Retry logic
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        endpoint,
                        headers={
                            "X-API-Key": self.api_key,
                            "Content-Type": "application/json",
                        },
                        json={
                            "texts": batch,
                            "normalize": True
                        },
                        verify=False,
                        timeout=200,
                    )
                    response.raise_for_status()
                    data = response.json()
                    embeddings = data if isinstance(data, list) else data.get("embeddings", [])
                    all_embeddings.extend(embeddings)
                    
                    time.sleep(0.5)
                    break  # Success
                    
                except requests.exceptions.HTTPError as e:
                    if attempt < self.max_retries - 1:
                        wait_time = self.initial_retry_delay * (2 ** attempt)
                        print(f"[WARN] HTTP {e.response.status_code} - Batch {i//batch_size + 1}, "
                              f"retry {attempt + 1}/{self.max_retries} in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"[ERROR] Failed after {self.max_retries} attempts")
                        raise
                        
                except requests.exceptions.RequestException as e:
                    if attempt < self.max_retries - 1:
                        wait_time = self.initial_retry_delay * (2 ** attempt)
                        print(f"[WARN] Request error: {type(e).__name__} - "
                              f"Retry {attempt + 1}/{self.max_retries} in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"[ERROR] Network error after {self.max_retries} attempts")
                        raise
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def build_embedding_model(ollama_base_url, model_name="bge", bge_api_url="", bge_api_key=""): 
    if model_name == "nomic":
        embedding_model = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=ollama_base_url
        )
    elif model_name == "e5":
        embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/e5-base-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
    elif model_name == "all-mpnet":
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
    elif model_name == "bge":
        embedding_model = BGEEmbeddings(
            api_url=bge_api_url,
            api_key=bge_api_key,
            model="BAAI/bge-m3"
        )

    return embedding_model


def build_qdrant_client(qdrant_url): 
    qdrant_client = QdrantClient(url=qdrant_url)
    return qdrant_client


def build_vectorstores(qdrant_client, embedding_model, qdrant_url, collection_names):
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
                embedding=embedding_model,
                collection_name=name,
                url=qdrant_url,
            )
            vectorstores[name] = store
        except Exception as e:
            print(f"[WARN] Errore su {name}: {e}")

    if not vectorstores:
        raise RuntimeError("Nessuna collezione valida trovata in Qdrant. Controlla i nomi in .env!")
    
    return vectorstores


def test_connection(vectorstores, embedding_model):
    if not vectorstores:
        print("Nessuna collezione valida trovata.")
        return False

    vec = embedding_model.embed_query("test")
    ok = False

    for name, store in vectorstores.items():
        try:
            docs = store.max_marginal_relevance_search_by_vector(
                vec, k=3, fetch_k=10, lambda_mult=0.5
            )
            print(f"[OK] Connessione riuscita per {name} ({len(docs)} risultati, su max 3).")
            ok = True
        except Exception as e:
            print(f"[WARN] Test fallito su {name}: {e}")

    return ok


def build_llm(model_name: str, ollama_base_url: str, creds, vllm_api_url, vllm_api_key):
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

    elif model_name == "vllm":
        llm = VLLMChat(
            api_url=vllm_api_url,
            api_key=vllm_api_key
        )
        
    return llm


def init_components(embedding_model_name: str, llm_model_name: str):
    OLLAMA_BASE_URL, QDRANT_URL, COLLECTION_NAMES, VLLM_API_URL, VLLM_API_KEY, BGE_API_URL, BGE_API_KEY = load_env_config()
    creds = load_google_creds()
    embedding_model = build_embedding_model(OLLAMA_BASE_URL, model_name=embedding_model_name, bge_api_url=BGE_API_URL, bge_api_key=BGE_API_KEY)
    qdrant_client = build_qdrant_client(QDRANT_URL)
    vectorstores = build_vectorstores(qdrant_client, embedding_model, QDRANT_URL, COLLECTION_NAMES)
    llm = build_llm(llm_model_name, OLLAMA_BASE_URL, creds, VLLM_API_URL, VLLM_API_KEY)

    return embedding_model, vectorstores, llm, COLLECTION_NAMES, qdrant_client