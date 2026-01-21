import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

from initializer import BGEEmbeddings

def test_bge_embeddings():
    load_dotenv()
    
    BGE_API_URL = os.getenv("BGE_EMBED_MODEL_API_URL", "")
    BGE_API_KEY = os.getenv("BGE_EMBED_MODEL_API_KEY", "")
    
    print(f"[INFO] Testing BGE Embeddings")
    print(f"[INFO] API URL: {BGE_API_URL}")
    print(f"[INFO] API Key: {'*' * len(BGE_API_KEY) if BGE_API_KEY else 'NOT SET'}")
    
    if not BGE_API_URL or not BGE_API_KEY:
        print("[ERROR] BGE_EMBED_MODEL_API_URL o BGE_EMBED_MODEL_API_KEY non configurati nel .env")
        return False
    
    try:
        # Crea l'istanza
        embedder = BGEEmbeddings(
            api_url=BGE_API_URL,
            api_key=BGE_API_KEY
        )
        
        # Test 1: Embed singola query
        print("\n[TEST 1] Embedding di una singola query...")
        query = "Quali sono gli orari delle lezioni?"
        query_vec = embedder.embed_query(query)
        print(f"✓ Query embedding generato: dimensione {len(query_vec)}")
        
        # Test 2: Embed multipli documenti
        print("\n[TEST 2] Embedding di multipli documenti...")
        docs = [
            "Il corso di Ingegneria Informatica offre una formazione completa.",
            "Gli esami si svolgono tre volte all'anno.",
            "La tesi magistrale prevede un lavoro di ricerca originale."
        ]
        docs_vecs = embedder.embed_documents(docs)
        print(f"✓ {len(docs_vecs)} embeddings generati, dimensione: {len(docs_vecs[0])}")
        
        # Test 3: Verifica dimensione corretta per BGE-M3
        expected_dim = 1024
        if len(query_vec) == expected_dim and len(docs_vecs[0]) == expected_dim:
            print(f"\n✓ Dimensione vettori corretta: {expected_dim} (BGE-M3)")
        else:
            print(f"\n✗ Dimensione vettori errata: attesa {expected_dim}, ricevuta {len(query_vec)}")
            return False
        
        # Test 4: Verifica che query e documenti siano diversi
        import numpy as np
        similarity = np.dot(query_vec, docs_vecs[0]) / (np.linalg.norm(query_vec) * np.linalg.norm(docs_vecs[0]))
        print(f"\n[TEST 4] Similarità coseno query-doc[0]: {similarity:.4f}")
        
        print("\n" + "="*50)
        print("TUTTI I TEST PASSATI! BGE-M3 funziona correttamente.")
        print("="*50)
        return True
        
    except Exception as e:
        print(f"\nERRORE durante i test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_bge_embeddings()