# SEMANTIC CHUNKING - indexing.py

import os, asyncio, uuid, argparse, json
import numpy as np
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


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic chunking indexing script")
    parser.add_argument("--embedding-model", type=str, default="nomic",
                        choices=["nomic", "e5", "all-mpnet", "bge"],
                        help="Seleziona il modello di embedding da usare")

    parser.add_argument("--sim-threshold", type=float, default=0.60,
                        help="Soglia similarità coseno per unire unità consecutive nello stesso chunk")
    parser.add_argument("--max-chars", type=int, default=4000,
                        help="Massimo numero di caratteri per chunk (vincolo pratico)")
    parser.add_argument("--min-chars", type=int, default=350,
                        help="Minimo numero di caratteri target: i chunk troppo piccoli vengono fusi se possibile")

    return parser.parse_args()


def build_embedding_model(args, OLLAMA_BASE_URL, BGE_API_URL, BGE_API_KEY):
    vector_size = 768

    if args.embedding_model == "nomic":
        embed = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    elif args.embedding_model == "e5":
        embed = HuggingFaceEmbeddings(
            model_name="intfloat/e5-base-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
    elif args.embedding_model == "all-mpnet":
        embed = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
    elif args.embedding_model == "bge":
        embed = BGEEmbeddings(api_url=BGE_API_URL, api_key=BGE_API_KEY, model="BAAI/bge-m3")
        vector_size = 1024
    else:
        raise ValueError("Unsupported embedding model")

    return embed, vector_size


def build_vectorstore(collection_name: str, embedding_model, vector_size: int, QDRANT_URL: str):
    client = QdrantClient(url=QDRANT_URL)

    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _mean_embedding(vectors: list[np.ndarray]) -> np.ndarray:
    if len(vectors) == 1:
        return vectors[0]
    return np.mean(np.stack(vectors, axis=0), axis=0)


def semantic_chunk_documents(
    docs: list[Document],
    embedding_model,
    sim_threshold: float = 0.78,
    max_chars: int = 1400,
    min_chars: int = 350,
    e5_prefix: bool = False
) -> list[Document]:

    unit_texts = []
    for d in docs:
        t = d.page_content.strip()
        if not t:
            unit_texts.append("")
            continue
        
        header = d.metadata.get("header")
        if header:
            t = f"{header}\n{t}"
        
        if e5_prefix:
            t = "passage: " + t
        unit_texts.append(t)

    keep_idx = [i for i, t in enumerate(unit_texts) if t.strip()]
    if not keep_idx:
        return []

    filtered_docs = [docs[i] for i in keep_idx]
    filtered_texts = [unit_texts[i] for i in keep_idx]

    try:
        vectors = embedding_model.embed_documents(filtered_texts)
    except Exception:
        vectors = [embedding_model.embed_query(t) for t in filtered_texts]

    vectors = [np.array(v, dtype=np.float32) for v in vectors]

    chunks: list[dict] = []
    cur_texts: list[str] = []
    cur_vecs: list[np.ndarray] = []
    cur_meta_ref: Document | None = None
    cur_headers: list[str | None] = []

    def flush():
        nonlocal cur_texts, cur_vecs, cur_meta_ref, cur_headers
        if not cur_texts:
            return
        
        final_parts = []
        last_header = None
        
        for i, (header, text) in enumerate(zip(cur_headers, cur_texts)):
            clean_text = text
            
            if e5_prefix and clean_text.startswith("passage: "):
                clean_text = clean_text[len("passage: "):]
            
            if header:
                header_with_newline = f"{header}\n"
                if clean_text.startswith(header_with_newline):
                    clean_text = clean_text[len(header_with_newline):]
            
            if header and header != last_header:
                final_parts.append(header)
                last_header = header
            
            if clean_text.strip():
                final_parts.append(clean_text)
        
        chunks.append({
            "text": "\n".join(final_parts).strip(),
            "vecs": cur_vecs[:],
            "meta_ref": cur_meta_ref
        })
        cur_texts, cur_vecs, cur_meta_ref, cur_headers = [], [], None, []

    for d, t, v in zip(filtered_docs, filtered_texts, vectors):
        header = d.metadata.get("header")
        
        if cur_meta_ref is None:
            cur_meta_ref = d

        if not cur_texts:
            cur_texts = [t]
            cur_vecs = [v]
            cur_headers = [header]
            continue

        cur_emb = _mean_embedding(cur_vecs)
        sim = _cosine(cur_emb, v)

        prospective_len = len("\n".join(cur_texts)) + 1 + len(t)

        if sim >= sim_threshold and prospective_len <= max_chars:
            cur_texts.append(t)
            cur_vecs.append(v)
            cur_headers.append(header)
        else:
            flush()
            cur_meta_ref = d
            cur_texts = [t]
            cur_vecs = [v]
            cur_headers = [header]

    flush()

    if len(chunks) > 1:
        merged = []
        i = 0
        while i < len(chunks):
            c = chunks[i]
            if len(c["text"]) >= min_chars or i == len(chunks) - 1:
                merged.append(c)
                i += 1
                continue

            nxt = chunks[i + 1]
            combined_text = (c["text"] + "\n" + nxt["text"]).strip()
            if len(combined_text) <= max_chars:
                merged.append({
                    "text": combined_text,
                    "vecs": c["vecs"] + nxt["vecs"],
                    "meta_ref": c["meta_ref"],
                })
                i += 2
            else:
                merged.append(c)
                i += 1
        chunks = merged

    final_docs: list[Document] = []
    for idx, c in enumerate(chunks):
        base_doc = c["meta_ref"]
        source_url = base_doc.metadata.get("source_url", "unknown")

        content = c["text"]

        final_docs.append(Document(
            page_content=content,
            metadata={
                **base_doc.metadata,
                "element_type": "SemanticChunk",
                "chunking": "semantic",
                "doc_id": sha(f"{source_url}_semantic_{idx}"),
                "num_units": len(c["vecs"]),
            }
        ))

    print(f"[INFO] Creati {len(final_docs)} semantic-chunk da {final_docs[0].metadata.get('source_url', 'unknown')}")
    return final_docs




def parse_links_file(LINKS_FILE: Path) -> dict[str, list[tuple[str, dict]]]:
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


def indexing_json_collection(collection_name: str, json_files: list[str], description_prefix: str,
                            build_vs_fn):
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
                "element_type": "JsonDoc"
            },
        )
        docs.append(doc)

    if not docs:
        print(f"[ERRORE] Nessun file valido per la collezione '{collection_name}'.")
        return

    vs = build_vs_fn(collection_name)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.metadata["doc_id"])) for doc in docs]
    vs.add_documents(docs, ids=ids)
    print(f"[OK] Inseriti {len(docs)} documenti JSON nella collezione '{collection_name}'")


async def process_block(collection_name: str, urls_with_opts: list[tuple[str, dict]],
                        embedding_model, vector_size, QDRANT_URL,
                        sim_threshold, max_chars, min_chars, e5_prefix):
    print(f"\n\n=== [BLOCCO: {collection_name}] ===")
    print(f"[INFO] Creazione/aggiornamento collezione: {collection_name}")

    all_units: list[Document] = []
    for url, opts in urls_with_opts:
        print(f"\n>>> Processando: {url} (depth={opts['depth']}, pdf={opts['pdf']})")
        units = await crawl(url, opts["depth"], opts["pdf"])
        all_units.extend(units)

    if not all_units:
        print(f"[WARN] Nessuna unità estratta per il blocco {collection_name}")
        return

    chunks = semantic_chunk_documents(
        docs=all_units,
        embedding_model=embedding_model,
        sim_threshold=sim_threshold,
        max_chars=max_chars,
        min_chars=min_chars,
        e5_prefix=e5_prefix
    )

    if not chunks:
        print(f"[WARN] Nessun chunk valido per il blocco {collection_name}")
        return

    vs = build_vectorstore(collection_name, embedding_model, vector_size, QDRANT_URL)

    ids = []
    for i, c in enumerate(chunks):
        base_id = sha(c.metadata.get("source_url", "unknown"))
        content_id = sha(c.page_content)
        chunk_id = f"{base_id}_semantic_{i}_{content_id[:8]}"
        c.metadata["chunk_id"] = chunk_id
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)))

    vs.add_documents(chunks, ids=ids)
    print(f"[OK] Upsert completato: {len(chunks)} punti inseriti nella collezione '{collection_name}'")


async def main():
    blocks = parse_links_file(LINKS_FILE)
    if not blocks:
        print("[ERRORE] Nessun blocco trovato in links.txt")
        return

    print(f"[INFO] Trovati {len(blocks)} blocchi nel file links.txt")

    embedding_model, vector_size = build_embedding_model(args, OLLAMA_BASE_URL, BGE_API_URL, BGE_API_KEY)

    e5_prefix = (args.embedding_model == "e5")

    for collection_name, urls_with_opts in blocks.items():
        await process_block(
            collection_name=collection_name,
            urls_with_opts=urls_with_opts,
            embedding_model=embedding_model,
            vector_size=vector_size,
            QDRANT_URL=QDRANT_URL,
            sim_threshold=args.sim_threshold,
            max_chars=args.max_chars,
            min_chars=args.min_chars,
            e5_prefix=e5_prefix
        )


if __name__ == "__main__":
    args = parse_args()
    load_dotenv()

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
    BGE_API_URL = os.getenv("BGE_EMBED_MODEL_API_URL", "")
    BGE_API_KEY = os.getenv("BGE_EMBED_MODEL_API_KEY", "")

    if args.embedding_model == "nomic":
        EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
    elif args.embedding_model == "e5":
        EMBED_MODEL = "intfloat/e5-base-v2"
    elif args.embedding_model == "all-mpnet":
        EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
    elif args.embedding_model == "bge":
        EMBED_MODEL = "BAAI/bge-m3"

    LINKS_FILE = Path(__file__).resolve().parent / "links.txt"

    asyncio.run(main())

    def _build_vs(name: str):
        embedding_model, vector_size = build_embedding_model(args, OLLAMA_BASE_URL, BGE_API_URL, BGE_API_KEY)
        return build_vectorstore(name, embedding_model, vector_size, QDRANT_URL)

    indexing_json_collection(
        collection_name="ing_info_mag_calendario_esami",
        json_files=["tab-calendario-esami.json"],
        description_prefix="Calendario appelli Ingegneria Informatica e Robotica 2025-26",
        build_vs_fn=_build_vs
    )

    indexing_json_collection(
        collection_name="ing_info_mag_regolamenti_didattici_tabelle",
        json_files=[
            "tab-regolamento-data-science.json",
            "tab-regolamento-data-science-2024.json",
            "tab-regolamento-robotics.json",
            "tab-regolamento-robotics-2024.json",
        ],
        description_prefix="Regolamento didattico",
        build_vs_fn=_build_vs
    )

    indexing_json_collection(
        collection_name="ing_info_mag_orari",
        json_files=["tab-orari-1-anno.json", "tab-orari-2-anno.json"],
        description_prefix="Orari lezioni",
        build_vs_fn=_build_vs
    )
