# SEMANTIC CHUNKING - parsing.py

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


# --- Semantic Chunking v2 ---
def semantic_chunk(elements, source_url, page_title, doc_type, file_name=None):
    from unstructured.documents.elements import Title, Header, NarrativeText, ListItem, Text, Table

    def word_count(s: str) -> int:
        return len(s.split())

    MAX_WORDS = 180          # NEW: soglia split chunk lunghi (tune 140–220)
    MIN_MERGE_WORDS = 40     # NEW: merge frammenti troppo corti
    crawl_ts = datetime.now(timezone.utc).isoformat()

    docs = []
    merged_chunks = []

    # NEW: teniamo uno stack di header per costruire il breadcrumb
    section_stack = []   # es: ["Regolamento Didattico", "Art. 3 Requisiti di accesso", "Comma 2"]
    current_buffer = []  # lista di paragrafi sotto la stessa sezione
    has_table_flag = False

    def current_section_path():
        return " / ".join([p for p in section_stack if p])

    def flush_buffer(force=False):
        """Converte il buffer in uno o più chunk compatti, con split se troppo lunghi."""
        nonlocal current_buffer, has_table_flag
        if not current_buffer:
            return
        # Merge dei paragrafi troppo corti
        tmp = []
        acc = ""
        for part in current_buffer:
            if not acc:
                acc = part
            elif word_count(acc) < MIN_MERGE_WORDS:
                acc = acc + "\n" + part
            else:
                tmp.append(acc)
                acc = part
        if acc:
            tmp.append(acc)

        # Split se troppo lunghi
        for t in tmp:
            if word_count(t) <= MAX_WORDS:
                merged_chunks.append((t, current_section_path(), has_table_flag))
            else:
                # split per frasi/righe approssimato
                sentences = [s.strip() for s in t.replace("\r", " ").split("\n") if s.strip()]
                acc2 = ""
                for s in sentences:
                    candidate = (acc2 + " " + s).strip() if acc2 else s
                    if word_count(candidate) > MAX_WORDS:
                        if acc2:
                            merged_chunks.append((acc2, current_section_path(), has_table_flag))
                        acc2 = s
                    else:
                        acc2 = candidate
                if acc2:
                    merged_chunks.append((acc2, current_section_path(), has_table_flag))

        # reset
        current_buffer = []
        has_table_flag = False

    # Passaggio sugli elementi unstructured
    for el in elements:
        text = getattr(el, "text", None)
        if not text:
            continue
        text = text.replace("\n", " ").strip()
        
        # --- Normalizzazione aggiuntiva per date e numeri ---
        import re
        text = re.sub(r'(\d{1,2}\s+[a-zàéìòù]+)', r'\1 |', text)  # separa pattern tipo '29 maggio'
        text = re.sub(r'(\d{4})', r'\1 |', text)  # separa anni

        if not text:
            continue

        # --- TITOLI/HEADER: aggiorna breadcrumb e flush buffer precedente
        if isinstance(el, (Title, Header)):
            flush_buffer()
            # euristica: manteniamo profondità max 3
            # non avendo il livello, trattiamo tutti gli Header come livello successivo
            # se la lista è già lunga, facciamo scorrere (H1/H2/H3)
            if len(section_stack) == 0:
                section_stack = [text]
            elif len(section_stack) == 1:
                section_stack = [section_stack[0], text]
            elif len(section_stack) == 2:
                section_stack = [section_stack[0], section_stack[1], text]
            else:
                # se compaiono tanti header consecutivi, comportati come nuova sottosezione
                section_stack = [section_stack[0], section_stack[1], text]
            continue

        # --- TESTO (paragrafi/liste)
        if isinstance(el, (NarrativeText, ListItem, Text)):
            # prepend header path nel contenuto per densificare segnale semantico
            header_line = current_section_path() or page_title or ""
            if header_line:
                current_buffer.append(f"{header_line}\n{text}")
            else:
                current_buffer.append(text)
            continue

        # --- TABELLE: allegale alla sezione corrente e marca flag
        if isinstance(el, Table):
            # se c'è testo accumulato prima, flush
            # poi aggiungi una entry specifica per tabella con contesto
            flush_buffer()
            header_line = current_section_path() or page_title or ""
            table_text = f"{header_line}\n[TABELLA]\n{text}" if header_line else f"[TABELLA]\n{text}"
            current_buffer.append(table_text)
            has_table_flag = True
            flush_buffer()  # la tabella rimane chunk singolo con suo contesto
            continue

    # flush finale
    flush_buffer(force=True)

    # Costruzione Document con metadata arricchiti
    for idx, (chunk_text, section_path, had_table) in enumerate(merged_chunks):
        docs.append(Document(
            page_content=chunk_text,
            metadata={
                "source_url": source_url,
                "doc_type": doc_type,
                "page_title": page_title,
                "file_name": file_name,
                "element_type": "SemanticChunk",
                "lang": ["ita"],
                "crawl_ts": crawl_ts,
                "section_path": section_path,          # NEW
                "has_table": had_table,                # NEW
                "doc_id": sha(f"{source_url}|{section_path}|{idx}"),  # NEW
            },
        ))

    print(f"[INFO] {len(docs)} chunk semantici creati da {source_url} (v2)")
    return docs



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

    return semantic_chunk(elements, source_url, page_title, doc_type="html", file_name=file_path.name)


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
    
    # --- PATCH: normalizza layout PDF tabellari per migliorare chunk semantico ---
    from unstructured.documents.elements import Text
    joined_text = " ".join(getattr(e, "text", "") for e in elements if getattr(e, "text", None))
    # se il PDF contiene molte cifre o pattern di date, lo riformatta come testo unico
    if sum(c.isdigit() for c in joined_text) > 20:
        print(f"[INFO] {file_path.name}: rilevato layout tabellare, normalizzazione testuale")
        elements = [Text(joined_text)]

    
    if not elements:
        print(f"[WARN] Nessun elemento estratto da {file_path}")
        return []

    return semantic_chunk(elements, source_url, page_title=file_path.stem, doc_type="pdf", file_name=file_path.name)