from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Literal, List, Optional
import uvicorn
import argparse
import signal
import sys
import json
import asyncio
import threading

from initializer import init_components, test_connection
from retrieval import build_bm25, build_corpus, build_spacy_tokenizer
from conversation_memory import ConversationMemory
from rag_graph import build_rag_graph

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
    session_id: Optional[str] = None
    force_fallback: bool = False
    stream: bool = False

class QueryResponse(BaseModel):
    status: str = "ok"
    answer: str
    contexts: List[str]
    search_technique: str

config = {
    "llm_model": "vllm",
    "embedding_model": "bge",
    "use_reranking": True,
    "rerank_method": "cross_encoder"
}
components = {}

rag_graph = build_rag_graph()

SESSION_MEMORIES: Dict[str, ConversationMemory] = {}

def parse_args():
    parser = argparse.ArgumentParser(description="Server API per sistema Q&A RAG")
    parser.add_argument("--llm-model", type=str, default="vllm",
                        choices=["llama-local", "gemini", "llama-api", "vllm"],
                        help="Modello LLM da utilizzare")
    parser.add_argument("--embedding-model", type=str, default="bge",
                        choices=["nomic", "e5", "all-mpnet", "bge"],
                        help="Modello di embedding da utilizzare")
    parser.add_argument("--reranking", action="store_true", default=True,
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
        
        embedding_model, vectorstores, llm, COLLECTION_NAMES, qdrant_client, reranker = init_components(
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
        components["reranker"] = reranker
        
        if config["use_reranking"] and config["rerank_method"] == "cross_encoder":
            if reranker:
                print("[INFO] Reranker BGE API disponibile")
            else:
                print("[WARN] Reranker BGE API non disponibile")
        
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

@app.post("/query")
async def process_query(request: QueryRequest):
    print(f"[DEBUG] Query ricevuta: {request.question}")
    print(f"[DEBUG] Search: {request.search_technique}, Reranking: {config['use_reranking']} ({config['rerank_method']})")
    print(f"[DEBUG] force_fallback={request.force_fallback}, stream={request.stream}")

    if not components:
        raise HTTPException(status_code=503, detail="Sistema non inizializzato")
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="La domanda non può essere vuota")
    
    sid = request.session_id or "default"
    mem = SESSION_MEMORIES.get(sid)
    if mem is None:
        mem = ConversationMemory(max_turns=4)
        SESSION_MEMORIES[sid] = mem
    memory_context = mem.get_context()

    if request.stream:
        return StreamingResponse(
            stream_query_events(request, sid, memory_context, mem),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    else:
        return await process_query_sync(request, sid, memory_context, mem)


async def stream_query_events(request: QueryRequest, sid: str, memory_context: str, mem: ConversationMemory):
    async_event_queue: asyncio.Queue = asyncio.Queue()
    result_container = {}
    error_container = {}
    heartbeat_interval = 30
    
    loop = asyncio.get_running_loop()

    def emit(event):
        try:
            loop.call_soon_threadsafe(async_event_queue.put_nowait, event)
        except RuntimeError:
            pass
    
    def run_graph():
        try:
            result = rag_graph.invoke({
                "question": request.question,
                "session_id": sid,
                "memory_context": memory_context,
                "search_technique": request.search_technique,
                "force_fallback": request.force_fallback,
                "llm": components["llm"],
                "embedding_model": components["embedding_model"],
                "embedding_model_name": config["embedding_model"],
                "vectorstores": components["vectorstores"],
                "corpus": components["corpus"],
                "bm25": components["bm25"],
                "nlp": components["spacy_tokenizer"],
                "use_reranking": config["use_reranking"],
                "rerank_method": config["rerank_method"],
                "reranker": components.get("reranker"),
                "emit": emit,
            })
            result_container["result"] = result
        except Exception as e:
            print(f"[ERROR] Errore elaborazione query: {str(e)}")
            import traceback
            traceback.print_exc()
            error_container["error"] = str(e)
        finally:
            try:
                loop.call_soon_threadsafe(async_event_queue.put_nowait, None)
            except RuntimeError:
                pass

    
    # Avvia il grafo in un thread separato
    thread = threading.Thread(target=run_graph, daemon=True, name=f"RAGThread-{sid}")
    thread.start()
    
    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    async_event_queue.get(), 
                    timeout=heartbeat_interval
                )
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                continue
            
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    
    except asyncio.CancelledError:
        print(f"[WARN] Stream cancellato per sessione {sid}")
        raise
    
    if "error" in error_container:
        error_event = {
            "type": "error",
            "message": error_container["error"]
        }
        yield f"data: {json.dumps(error_event)}\n\n"
    elif "result" in result_container:
        result = result_container["result"]
        answer = result["answer"]
        contexts = result.get("contexts", [])
        mem.add_turn(request.question, answer)
        
        final_event = {
            "type": "result",
            "status": "ok",
            "answer": answer,
            "contexts": contexts,
            "search_technique": request.search_technique
        }
        yield f"data: {json.dumps(final_event, ensure_ascii=False)}\n\n"
    else:
        error_event = {
            "type": "error",
            "message": "Timeout o errore sconosciuto nell'elaborazione"
        }
        yield f"data: {json.dumps(error_event)}\n\n"


async def process_query_sync(request: QueryRequest, sid: str, memory_context: str, mem: ConversationMemory):
    try:
        result = rag_graph.invoke({
            "question": request.question,
            "session_id": sid,
            "memory_context": memory_context,
            "search_technique": request.search_technique,
            "force_fallback": request.force_fallback,
            "llm": components["llm"],
            "embedding_model": components["embedding_model"],
            "embedding_model_name": config["embedding_model"],
            "vectorstores": components["vectorstores"],
            "corpus": components["corpus"],
            "bm25": components["bm25"],
            "nlp": components["spacy_tokenizer"],
            "use_reranking": config["use_reranking"],
            "rerank_method": config["rerank_method"],
            "reranker": components.get("reranker"),
        })

        answer = result["answer"]
        contexts = result.get("contexts", [])

        mem.add_turn(request.question, answer)
        
        return QueryResponse(
            status="ok",
            answer=answer,
            contexts=contexts,
            search_technique=request.search_technique,
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