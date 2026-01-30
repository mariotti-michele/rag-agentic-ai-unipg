# SECTION BASED LIMITED CHUNKING - parsing.py

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


def split_long_chunk(chunk_text: str, max_chars: int = 2000, overlap: int = 150) -> list[str]:
    """Divide un chunk troppo lungo in sotto-chunk con overlap"""
    if len(chunk_text) <= max_chars:
        return [chunk_text]
    
    # Estrai il titolo se presente (prima riga)
    lines = chunk_text.split('\n', 1)
    title = lines[0] if len(lines) > 1 and len(lines[0]) < 200 else ""
    content = lines[1] if len(lines) > 1 and title else chunk_text
    
    sub_chunks = []
    start = 0
    
    while start < len(content):
        # Calcola la fine del chunk corrente
        end = start + max_chars - len(title) - 1  # -1 per il newline
        
        if end >= len(content):
            # Ultimo chunk
            if title:
                sub_chunks.append(f"{title}\n{content[start:]}")
            else:
                sub_chunks.append(content[start:])
            break
        
        # Cerca di spezzare su un separatore naturale (newline, punto, virgola)
        search_start = max(start, end - 100)
        search_end = min(len(content), end + 100)
        
        best_split = end
        for separator in ['\n\n', '\n', '. ', ', ', ' ']:
            pos = content.rfind(separator, search_start, search_end)
            if pos != -1 and pos > start:  # Garantisci pos > start
                best_split = pos + len(separator)
                break
        
        # Se best_split non avanza rispetto a start, forza uno split minimo
        if best_split <= start:
            best_split = min(len(content), start + max(1, max_chars - len(title) - 1))
        
        # Aggiungi il chunk con il titolo
        if title:
            sub_chunks.append(f"{title}\n{content[start:best_split]}")
        else:
            sub_chunks.append(content[start:best_split])
        
        # Calcola prossimo start con overlap
        start_next = best_split - overlap
        
        # Se l'overlap non fa avanzare, disabilita overlap E garantisci avanzamento
        if start_next <= start:
            start_next = best_split
        
        start = start_next
    
    return sub_chunks


def chunk(elements, source_url, page_title, doc_type, file_name=None, max_chars=2000, overlap=150):
    from unstructured.documents.elements import Title, Header, NarrativeText, ListItem, Text, Table

    crawl_ts = datetime.now(timezone.utc).isoformat()
    docs, merged_chunks = [], []
    current_chunk, current_header = "", None

    for el in elements:
        text = getattr(el, "text", None)
        if not text:
            continue

        if isinstance(el, (Title, Header)):
            if current_chunk.strip():
                merged_chunks.append(current_chunk.strip())
                current_chunk = ""
            current_header = text.strip()
            continue

        if isinstance(el, (NarrativeText, ListItem, Text)):
            if current_header:
                current_chunk = f"{current_header}\n{text}"
                current_header = None
            else:
                current_chunk += ("\n" if current_chunk else "") + text
            continue

        if isinstance(el, Table):
            if current_chunk.strip():
                merged_chunks.append(current_chunk.strip())
                current_chunk = ""
            merged_chunks.append(f"[TABELLA]\n{text}")

    if current_chunk.strip():
        merged_chunks.append(current_chunk.strip())

    # Dividi i chunk troppo lunghi
    final_chunks = []
    for chunk_text in merged_chunks:
        sub_chunks = split_long_chunk(chunk_text, max_chars, overlap)
        final_chunks.extend(sub_chunks)

    # Crea i documenti
    for idx, chunk_text in enumerate(final_chunks):
        docs.append(Document(
            page_content=chunk_text,
            metadata={
                "source_url": source_url,
                "doc_type": doc_type,
                "page_title": page_title,
                "file_name": file_name,
                "element_type": "SectionBasedChunk",
                "lang": ["ita"],
                "crawl_ts": crawl_ts,
                "doc_id": sha(f"{source_url}_{idx}"),
                "chunk_length": len(chunk_text),
            },
        ))

    print(f"[INFO] {len(merged_chunks)} chunk iniziali -> {len(final_chunks)} chunk finali da {source_url}")
    return docs


def to_documents_from_html(file_path: Path, source_url: str, page_title: str) -> list[Document]:
    raw_html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")

    main_el = soup.find("main")
    if not main_el:
        print(f"[WARN] Nessun <main> trovato in {source_url}, salto")
        return []
    
    for mod in main_el.select("div.module-container.col-xs-12"):
        mod.decompose()

    main_html = str(main_el)
    
    elements = partition_html(
        text=main_html,
        include_page_breaks=False,
        languages=["ita", "eng"]
    )

    if not elements:
        text_fallback = main_el.get_text(separator="\n", strip=True)
        return [Document(
            page_content=text_fallback,
            metadata={
                "source_url": source_url,
                "doc_type": "html",
                "page_title": page_title,
                "element_type": "FallbackText",
                "lang": "ita",
                "crawl_ts": datetime.now(timezone.utc).isoformat(),
                "doc_id": sha(source_url),
            },
        )] if text_fallback else []

    return chunk(elements, source_url, page_title, doc_type="html", file_name=file_path.name)


def to_documents_from_pdf(file_path: Path, source_url: str) -> list[Document]:
    try:
        elements = partition_pdf(
            filename=str(file_path),
            strategy="hi_res",
            include_page_breaks=True,
            infer_table_structure=True,
            languages=["ita", "eng"]
        )
    except Exception as e:
        print(f"[WARN] partition_pdf fallito su {file_path}: {e}")
        return []
    
    if not elements:
        print(f"[WARN] Nessun elemento estratto da {file_path}")
        return []

    return chunk(elements, source_url, page_title=file_path.stem, doc_type="pdf", file_name=file_path.name)