# SECTION BASED LIMITED CHUNKING INDEXING SCRIPT

import os, asyncio, uuid
from dotenv import load_dotenv
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from crawling import crawl
from scraping import sha
from bge_embedding_class import BGEEmbeddings

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Section based chunking indexing script")
    parser.add_argument("--embedding-model", type=str, default="bge", 
                        choices=["nomic", "e5", "all-mpnet", "bge"],
                        help="Seleziona il modello di embedding da usare")
    args = parser.parse_args()
    return args


def chunk_documents(docs: list[Document]) -> list[Document]:
    clean_chunks = []
    for i, d in enumerate(docs):
        text = d.page_content.strip()
        if(args.embedding_model == "e5"):
            text = "passage: " + text
            d.page_content = text
        if not text:
            continue
        base_id = sha(d.metadata["source_url"])
        content_id = sha(d.page_content)
        d.metadata["chunk_id"] = f"{base_id}_{i}_{content_id[:8]}"
        clean_chunks.append(d)

    print(f"[INFO] Mantieni {len(clean_chunks)} chunk (nessun text split)")
    return clean_chunks



def build_vectorstore(collection_name: str):
    embedding_model = None
    vector_size = 768
    
    if args.embedding_model == "nomic":
        embedding_model = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    elif args.embedding_model == "e5":
        embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/e5-base-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
    elif args.embedding_model == "all-mpnet":
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
    elif args.embedding_model == "bge":
        embedding_model = BGEEmbeddings(
            api_url=BGE_API_URL,
            api_key=BGE_API_KEY,
            model="BAAI/bge-m3"
        )
        vector_size = 1024
        
    client = QdrantClient(url=QDRANT_URL)

    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name not in existing_collections:
        print(f"[INFO] Creazione nuova collezione Qdrant: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": vector_size, "distance": "Cosine"},
        )
    else:
        print(f"[INFO] Collezione già esistente: {collection_name}")

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )


def parse_links_file() -> dict[str, list[tuple[str, dict]]]:
    blocks = {}
    current_block = None

    for line in LINKS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("#"):
            current_block = line.lstrip("#").strip()
            sanitized_name = (
                current_block.replace(" ", "_")
                .replace("/", "_")
                .replace("-", "_")
                .replace(".", "_")
            )
            blocks[sanitized_name] = []
        else:
            if current_block is None:
                continue

            parts = line.split()
            url = parts[0]
            options = {"depth": 0, "pdf": True}

            if "--no-pdf" in parts:
                options["pdf"] = False
            for i, p in enumerate(parts):
                if p == "--depth" and i + 1 < len(parts):
                    options["depth"] = int(parts[i + 1])

            blocks[sanitized_name].append((url, options))

    return blocks


async def process_block(collection_name: str, urls_with_opts: list[tuple[str, dict]]):
    print(f"\n\n=== [BLOCCO: {collection_name}] ===")
    print(f"[INFO] Creazione/aggiornamento collezione: {collection_name}")

    all_docs = []
    for url, opts in urls_with_opts:
        print(f"\n>>> Processando: {url} (depth={opts['depth']}, pdf={opts['pdf']})")
        docs = await crawl(url, opts["depth"], opts["pdf"])
        all_docs.extend(docs)

    chunks = chunk_documents(all_docs)
    print(f"[INFO] Chunks finali da inserire: {len(chunks)}")

    if not chunks:
        print(f"[WARN] Nessun chunk valido per il blocco {collection_name}")
        return

    vs = build_vectorstore(collection_name)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"])) for c in chunks]
    vs.add_documents(chunks, ids=ids)
    print(f"[OK] Upsert completato: {len(chunks)} punti inseriti nella collezione '{collection_name}'")


async def main():
    blocks = parse_links_file()
    if not blocks:
        print("[ERRORE] Nessun blocco trovato in links.txt")
        return

    print(f"[INFO] Trovati {len(blocks)} blocchi nel file links.txt")

    for collection_name, urls_with_opts in blocks.items():
        await process_block(collection_name, urls_with_opts)


import json

def indexing_json_collection(collection_name: str, json_files: list[str], description_prefix: str):
    base_path = Path(__file__).resolve().parent / "json-data"
    docs = []

    for filename in json_files:
        json_path = base_path / filename

        if not json_path.exists():
            print(f"[ERRORE] File non trovato: {json_path}")
            continue

        data = json.loads(json_path.read_text(encoding="utf-8"))
        text_content = json.dumps(data, ensure_ascii=False, indent=2)

        doc = Document(
            page_content=text_content,
            metadata={
                "source_url": "manual",
                "doc_type": "json",
                "description": f"{description_prefix} - {filename}",
                "doc_id": sha(filename),
            },
        )
        docs.append(doc)

    if not docs:
        print(f"[ERRORE] Nessun file valido per la collezione '{collection_name}'.")
        return

    vs = build_vectorstore(collection_name)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.metadata["doc_id"])) for doc in docs]
    vs.add_documents(docs, ids=ids)
    print(f"[OK] Inseriti {len(docs)} documenti JSON nella collezione '{collection_name}'")


if __name__ == "__main__":
    args = parse_args()

    load_dotenv()

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
    BGE_API_URL = os.getenv("BGE_EMBED_MODEL_API_URL", "")
    BGE_API_KEY = os.getenv("BGE_EMBED_MODEL_API_KEY", "")

    if(args.embedding_model == "nomic"):
        EMBED_MODEL = "nomic-embed-text"
    elif(args.embedding_model == "e5"):
        EMBED_MODEL = "intfloat/e5-base-v2"
    elif(args.embedding_model == "all-mpnet"):
        EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
    elif(args.embedding_model == "bge"):
        EMBED_MODEL = "BAAI/bge-m3"

    LINKS_FILE = Path(__file__).resolve().parent / "links.txt"

    asyncio.run(main())
    indexing_json_collection(
        collection_name="ing_info_mag_calendario_esami",
        json_files=["tab-calendario-esami.json"],
        description_prefix="Calendario appelli Ingegneria Informatica e Robotica 2025-26"
    )

    indexing_json_collection(
        collection_name="ing_info_mag_regolamenti_didattici_tabelle",
        json_files=[
            "tab-regolamento-data-science.json",
            "tab-regolamento-data-science-2024.json",
            "tab-regolamento-robotics.json",
            "tab-regolamento-robotics-2024.json",
        ],
        description_prefix="Regolamento didattico"
    )

    indexing_json_collection(
        collection_name="ing_info_mag_orari",
        json_files=["tab-orari-1-anno.json", "tab-orari-2-anno.json"],
        description_prefix="Orari lezioni"
    )

    indexing_json_collection(
        collection_name="ing_info_calendario_lauree",
        json_files=["tab-calendario-lauree.json"],
        description_prefix="Calendario lauree"
    )
