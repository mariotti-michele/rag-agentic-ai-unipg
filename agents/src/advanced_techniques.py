from typing import List, Dict
import numpy as np
from prompts import RERANK_PROMPT
from sentence_transformers import CrossEncoder


def rerank_documents(query: str, docs: List[Dict], llm, top_k: int = 5) -> List[Dict]:
    # non riegge a reggere 10 chunk completi -> contesto troppo grande, inoltre molto lento
    # usarne solo 5 non ha senso, non farei nessuna selezione ma solo riordinamento
    # quindi prendo solo i primi 300 char di ogni chunk 
    if not docs:
        return []
    
    if len(docs) <= top_k:
        return docs

    docs_text = ""
    for i, doc in enumerate(docs[:top_k * 2], 1):
        docs_text += f"[Doc {i}]\n{doc['text'][:300]}...\n\n"
    
    try:
        prompt = RERANK_PROMPT.format(query=query, documents=docs_text)
        response = llm.invoke(prompt)
        
        if hasattr(response, "content"):
            response = response.content
        
        scores = [float(s.strip()) for s in response.strip().split(",")]
        
        reranked_docs = []
        for i, doc in enumerate(docs[:len(scores)]):
            new_doc = doc.copy()
            new_doc["rerank_score"] = scores[i]
            new_doc["original_score"] = doc.get("score", 0)
            reranked_docs.append(new_doc)
        
        reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        print(f"[INFO] Re-ranking completato: {len(reranked_docs)} documenti valutati")
        return reranked_docs[:top_k]
        
    except Exception as e:
        print(f"[WARN] Errore nel re-ranking, uso ordine originale: {e}")
        return docs[:top_k]


def rerank_with_cross_encoder(query: str, docs: List[Dict], model_name: str = "BAAI/bge-reranker-v2-m3", top_k: int = 5) -> List[Dict]:
    try:
        if not docs:
            return []
        
        model = CrossEncoder(model_name)
        pairs = [[query, doc["text"]] for doc in docs]
        scores = model.predict(pairs)
        
        reranked_docs = []
        for i, doc in enumerate(docs):
            new_doc = doc.copy()
            new_doc["rerank_score"] = float(scores[i])
            new_doc["original_score"] = doc.get("score", 0)
            reranked_docs.append(new_doc)
        reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        print(f"[INFO] Cross-encoder re-ranking completato: {len(reranked_docs)} documenti")
        return reranked_docs[:top_k]
        
    except ImportError:
        print("[WARN] sentence-transformers non installato, uso ordine originale")
        return docs[:top_k]
    except Exception as e:
        print(f"[WARN] Errore nel cross-encoder re-ranking: {e}")
        return docs[:top_k]