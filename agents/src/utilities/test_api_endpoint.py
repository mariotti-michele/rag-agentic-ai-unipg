import requests
import os
from dotenv import load_dotenv

load_dotenv()

BGE_API_URL = os.getenv("BGE_EMBED_MODEL_API_URL", "")
BGE_API_KEY = os.getenv("BGE_EMBED_MODEL_API_KEY", "")

# Prova diversi endpoint
endpoints_to_test = [
    "https://141.250.40.120/embeddings",
    "https://141.250.40.120/v1/embeddings",
    "https://141.250.40.120/embed",
    "https://141.250.40.120/api/embeddings",
]

print("Testing different endpoints...\n")

for endpoint in endpoints_to_test:
    print(f"[TEST] {endpoint}")
    try:
        response = requests.post(
            endpoint,
            headers={
                "X-API-Key": BGE_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "model": "BAAI/bge-m3",
                "input": ["test"],
            },
            verify=False,
            timeout=10,
        )
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            print(f"  ✅ SUCCESS! Endpoint corretto: {endpoint}")
            print(f"  Response: {response.json()}")
            break
        else:
            print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    print()