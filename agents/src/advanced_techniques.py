from typing import List, Dict
import numpy as np


def rerank_documents(query: str, docs: List[Dict], llm, top_k: int = 3) -> List[Dict]:
    """
    Re-ranking dei documenti usando un LLM.
    
    Args:
        query: La query dell'utente
        docs: Lista di documenti con text, score, collection, etc.
        llm: Il modello LLM per il re-ranking
        top_k: Numero di documenti da restituire dopo il re-ranking
    
    Returns:
        Lista di documenti ri-ordinati per rilevanza
    """
    if not docs:
        return []
    
    if len(docs) <= top_k:
        return docs
    
    RERANK_PROMPT = """Dato questo contesto e la domanda dell'utente, assegna un punteggio di rilevanza da 0 a 10 a ciascun documento.
Rispondi SOLO con i numeri separati da virgole, nello stesso ordine dei documenti.

Domanda: {query}

Documenti:
{documents}

Punteggi (separati da virgole):"""

    # Prepara i documenti per il prompt
    docs_text = ""
    for i, doc in enumerate(docs[:top_k * 2], 1):  # Considera il doppio per avere margine
        docs_text += f"[Doc {i}]\n{doc['text'][:300]}...\n\n"
    
    try:
        prompt = RERANK_PROMPT.format(query=query, documents=docs_text)
        response = llm.invoke(prompt)
        
        if hasattr(response, "content"):
            response = response.content
        
        # Parsing dei punteggi
        scores = [float(s.strip()) for s in response.strip().split(",")]
        
        # Aggiorna i documenti con i nuovi punteggi
        reranked_docs = []
        for i, doc in enumerate(docs[:len(scores)]):
            new_doc = doc.copy()
            new_doc["rerank_score"] = scores[i]
            new_doc["original_score"] = doc.get("score", 0)
            reranked_docs.append(new_doc)
        
        # Ordina per nuovo punteggio
        reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        print(f"[INFO] Re-ranking completato: {len(reranked_docs)} documenti valutati")
        return reranked_docs[:top_k]
        
    except Exception as e:
        print(f"[WARN] Errore nel re-ranking, uso ordine originale: {e}")
        return docs[:top_k]


def rerank_with_cross_encoder(query: str, docs: List[Dict], model_name: str = "BAAI/bge-reranker-v2-m3", top_k: int = 3) -> List[Dict]:
    """
    Re-ranking usando un Cross-Encoder pre-addestrato.
    Più veloce del metodo LLM-based.
    
    Args:
        query: La query dell'utente
        docs: Lista di documenti
        model_name: Nome del modello cross-encoder
        top_k: Numero di documenti da restituire
    
    Returns:
        Lista di documenti ri-ordinati
    """
    try:
        from sentence_transformers import CrossEncoder
        
        if not docs:
            return []
        
        model = CrossEncoder(model_name)
        
        # Prepara le coppie query-documento
        pairs = [[query, doc["text"]] for doc in docs]
        
        # Calcola i punteggi
        scores = model.predict(pairs)
        
        # Aggiorna i documenti
        reranked_docs = []
        for i, doc in enumerate(docs):
            new_doc = doc.copy()
            new_doc["rerank_score"] = float(scores[i])
            new_doc["original_score"] = doc.get("score", 0)
            reranked_docs.append(new_doc)
        
        # Ordina per punteggio
        reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        print(f"[INFO] Cross-encoder re-ranking completato: {len(reranked_docs)} documenti")
        return reranked_docs[:top_k]
        
    except ImportError:
        print("[WARN] sentence-transformers non installato, uso ordine originale")
        return docs[:top_k]
    except Exception as e:
        print(f"[WARN] Errore nel cross-encoder re-ranking: {e}")
        return docs[:top_k]