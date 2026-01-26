import requests
from langchain_core.embeddings import Embeddings
from typing import List

class BGEEmbeddings(Embeddings):
    api_url: str
    api_key: str
    model: str = "BAAI/bge-m3"

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
        batch_size = 4  # Adjust based on your GPU memory
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            endpoint = self._endpoint()
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
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
