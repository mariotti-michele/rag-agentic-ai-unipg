# FIXED SIZE CHUNKING - parsing.py

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

    main_el = soup.find("main") or soup

    # Rimuove i moduli sopra al main
    for mod in main_el.select("div.module-container.col-xs-12"):
        mod.decompose()

    text = main_el.get_text(separator="\n", strip=True)
    if not text:
        print(f"[WARN] Nessun testo trovato in {source_url}")
        return []

    crawl_ts = datetime.now(timezone.utc).isoformat()
    doc = Document(
        page_content=text,
        metadata={
            "source_url": source_url,
            "doc_type": "html",
            "page_title": page_title,
            "element_type": "FullText",
            "lang": "ita",
            "crawl_ts": crawl_ts,
            "doc_id": sha(source_url),
        },
    )
    return [doc]


def to_documents_from_pdf(file_path: Path, source_url: str) -> list[Document]:
    docs = []
    crawl_ts = datetime.now(timezone.utc).isoformat()

    try:
        elements = partition_pdf(
            filename=str(file_path),
            strategy="hi_res",
            include_page_breaks=False,
            infer_table_structure=False,
            languages=["ita", "eng"],
        )
        text_all = "\n".join(
            getattr(el, "text", "").strip() for el in elements if getattr(el, "text", "").strip()
        )
    except Exception as e:
        print(f"[WARN] partition_pdf fallito su {file_path}: {e}")
        text_all = ""

    if not text_all.strip():
        print(f"[WARN] Nessun testo estratto da {file_path.name}")
        return []

    docs.append(Document(
        page_content=text_all.strip(),
        metadata={
            "source_url": source_url,
            "doc_type": "pdf",
            "file_name": file_path.name,
            "element_type": "FullText",
            "crawl_ts": crawl_ts,
            "doc_id": sha(source_url),
        },
    ))

    return docs