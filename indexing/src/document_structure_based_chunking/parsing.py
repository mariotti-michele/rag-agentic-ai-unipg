# DOCUMENT-STRUCTURE BASED CHUNKING - parsing.py

from bs4 import BeautifulSoup
from datetime import datetime, timezone
from pathlib import Path

from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document

from scraping import sha

import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="camelot")


def to_documents_from_html(file_path: Path, source_url: str, page_title: str) -> list[Document]:
    raw_html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")

    main_el = soup.find("main")
    if not main_el:
        print(f"[WARN] Nessun <main> trovato in {source_url}, salto")
        return []
    
    # Rimuove i moduli sopra al main
    for mod in main_el.select("div.module-container.col-xs-12"):
        mod.decompose()

    main_html = str(main_el)
   
    elements = partition_html(
        text=main_html,
        include_page_breaks=False,
        languages=["ita", "eng"]
    )

    docs = []
    crawl_ts = datetime.now(timezone.utc).isoformat()

    if not elements:
        print(f"[WARN] partition_html non ha trovato elementi in {source_url}, uso fallback BeautifulSoup")
        text_fallback = main_el.get_text(separator="\n", strip=True)
        if text_fallback:
            docs.append(Document(
                page_content=text_fallback,
                metadata={
                    "source_url": source_url,
                    "doc_type": "html",
                    "page_title": page_title,
                    "element_type": "FallbackText",
                    "lang": "ita",
                    "crawl_ts": crawl_ts,
                    "doc_id": sha(source_url),
                },
            ))
        return docs

    for el in elements:
        text = getattr(el, "text", None) or str(el).strip()
        if not text:
            continue

        element_type = getattr(el, "category", None) or el.__class__.__name__
        meta = getattr(el, "metadata", None)
        meta = meta.to_dict() if meta is not None else {}

        doc = Document(
            page_content=text,
            metadata={
                "source_url": source_url,
                "doc_type": "html",
                "page_title": page_title,
                "element_type": element_type,
                "lang": meta.get("languages", None),
                "crawl_ts": crawl_ts,
                "doc_id": sha(source_url),
            },
        )
        docs.append(doc)
    return docs


def to_documents_from_pdf(file_path: Path, source_url: str) -> list[Document]:
    docs = []
    crawl_ts = datetime.now(timezone.utc).isoformat()

    elements = []
    try:
        elements = partition_pdf(
            filename=str(file_path),
            strategy="hi_res",
            include_page_breaks=True,
            infer_table_structure=True,
            languages=["ita", "eng"]
        )
    except Exception as e:
        print(f"[WARN] partition_pdf fallito con strategy hi_res su {file_path}: {e}")

    for el in elements:
        text = getattr(el, "text", None) or str(el).strip()
        if not text:
            continue
        element_type = getattr(el, "category", None) or el.__class__.__name__
        meta = getattr(el, "metadata", None)
        meta = meta.to_dict() if meta is not None else {}

        docs.append(Document(
            page_content=text,
            metadata={
                "source_url": source_url,
                "doc_type": "pdf",
                "page_number": meta.get("page_number", None),
                "file_name": file_path.name,
                "element_type": element_type,
                "lang": meta.get("languages", None),
                "crawl_ts": crawl_ts,
                "doc_id": sha(source_url),
            },
        ))

    return docs