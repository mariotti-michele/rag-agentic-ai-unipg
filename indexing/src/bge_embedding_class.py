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
        if u.endswith("/v1"):
            return u.rstrip("/v1") + "/embeddings/embed"
        return u + "/embeddings/embed"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        endpoint = self._endpoint()
        response = requests.post(
            endpoint,
            headers={
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            },
            json={
                "texts": texts,
                "normalize": True
            },
            verify=False,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else data.get("embeddings", [])

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
