import requests
import time
from langchain_core.embeddings import Embeddings
from typing import List

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
        # Process in smaller batches to avoid OOM
        batch_size = 2  # Adjust based on your GPU memory
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
                    
                    # Delay tra batch
                    time.sleep(0.5)
                    break  # Success, esci dal retry loop
                    
                except requests.exceptions.HTTPError as e:
                    if attempt < self.max_retries - 1:
                        wait_time = self.initial_retry_delay * (2 ** attempt)
                        print(f"[WARN] HTTP {e.response.status_code} - Batch {i//batch_size + 1}, "
                              f"retry {attempt + 1}/{self.max_retries} in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"[ERROR] Failed after {self.max_retries} attempts for batch starting at {i}")
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
