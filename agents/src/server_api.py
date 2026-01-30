from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List
import uvicorn
import argparse
import signal
import sys

from initializer import init_components, test_connection
from query_processing import generate_answer
from retrieval import build_bm25, build_corpus, build_spacy_tokenizer

app = FastAPI(title="RAG Q&A API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in futuro specificare URL della Web app
    allow_credentials=False, # e rimetti True
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    search_technique: Literal["dense", "sparse", "hybrid"] = "dense"

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]
    mode: str
    search_technique: str

config = {
    "llm_model": "vllm",
    "embedding_model": "bge",
    "use_reranking": False,
    "rerank_method": "cross_encoder"
}
components = {}

def parse_args():
    parser = argparse.ArgumentParser(description="Server API per sistema Q&A RAG")
    parser.add_argument("--llm-model", type=str, default="vllm",
                        choices=["llama-local", "gemini", "llama-api", "vllm"],
                        help="Modello LLM da utilizzare")
    parser.add_argument("--embedding-model", type=str, default="bge",
                        choices=["nomic", "e5", "all-mpnet", "bge"],
                        help="Modello di embedding da utilizzare")
    parser.add_argument("--reranking", action="store_true",
                        help="Attiva il re-ranking dei documenti")
    parser.add_argument("--rerank-method", type=str, default="cross_encoder",
                        choices=["cross_encoder", "llm"],
                        help="Metodo di re-ranking: cross_encoder (veloce) o llm (accurato)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host per il server")
    parser.add_argument("--port", type=int, default=8000,
                        help="Porta per il server")
    return parser.parse_args()

@app.on_event("startup")
async def startup_event():
    try:
        print("Inizializzazione componenti RAG...")
        print(f"LLM Model: {config['llm_model']}")
        print(f"Embedding Model: {config['embedding_model']}")
        print(f"Reranking: {config['use_reranking']} (metodo: {config['rerank_method']})")
        
        embedding_model, vectorstores, llm, COLLECTION_NAMES, qdrant_client = init_components(
            embedding_model_name=config["embedding_model"],
            llm_model_name=config["llm_model"]
        )
        
        if not test_connection(vectorstores, embedding_model):
            raise RuntimeError("Impossibile connettersi al vector store")
        
        corpus, corpus_texts = build_corpus(qdrant_client, COLLECTION_NAMES)
        spacy_tokenizer = build_spacy_tokenizer()
        bm25 = build_bm25(corpus_texts, spacy_tokenizer)
        
        components["embedding_model"] = embedding_model
        components["vectorstores"] = vectorstores
        components["llm"] = llm
        components["COLLECTION_NAMES"] = COLLECTION_NAMES
        components["qdrant_client"] = qdrant_client
        components["corpus"] = corpus
        components["bm25"] = bm25
        components["spacy_tokenizer"] = spacy_tokenizer
        
        print("Sistema RAG inizializzato con successo")
        
    except Exception as e:
        print(f"Errore durante l'inizializzazione: {e}")
        raise

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "RAG Q&A API Server",
        "version": "1.0.0",
        "llm_model": config["llm_model"],
        "embedding_model": config["embedding_model"]
    }

@app.get("/health")
async def health_check():
    if not components:
        raise HTTPException(status_code=503, detail="Sistema non inizializzato")
    
    return {
        "status": "healthy",
        "components_loaded": len(components) > 0,
        "collections": components.get("COLLECTION_NAMES", []),
        "llm_model": config["llm_model"],
        "embedding_model": config["embedding_model"]
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    print(f"[DEBUG] Query ricevuta: {request.question[:50]}...")
    print(f"[DEBUG] Search: {request.search_technique}, Reranking: {config['use_reranking']} ({config['rerank_method']})")
    
    if not components:
        raise HTTPException(status_code=503, detail="Sistema non inizializzato")
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="La domanda non può essere vuota")
    
    try:
        answer, contexts, mode = generate_answer(
            llm=components["llm"],
            query=request.question,
            search_technique=request.search_technique,
            embedding_model=components["embedding_model"],
            embedding_model_name=config["embedding_model"],
            vectorstores=components["vectorstores"],
            corpus=components["corpus"],
            bm25=components["bm25"],
            nlp=components["spacy_tokenizer"],
            use_reranking=config["use_reranking"],
            rerank_method=config["rerank_method"]
        )
        
        return QueryResponse(
            answer=answer,
            contexts=contexts,
            mode=mode,
            search_technique=request.search_technique
        )

    except Exception as e:
        print(f"[ERROR] Errore elaborazione query: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore durante l'elaborazione: {str(e)}")

@app.get("/config")
async def get_config():
    return {
        "llm_model": config["llm_model"],
        "embedding_model": config["embedding_model"],
        "use_reranking": config["use_reranking"],
        "rerank_method": config["rerank_method"],
        "available_search_techniques": ["dense", "sparse", "hybrid"]
    }

def signal_handler(sig, frame):
    print("\nRicevuto segnale di terminazione. Chiusura graceful...")
    sys.exit(0)

if __name__ == "__main__":
    args = parse_args()
    
    config["llm_model"] = args.llm_model
    config["embedding_model"] = args.embedding_model
    config["use_reranking"] = args.reranking
    config["rerank_method"] = args.rerank_method
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    uvicorn.run(
        "server_api:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
        access_log=True
    )