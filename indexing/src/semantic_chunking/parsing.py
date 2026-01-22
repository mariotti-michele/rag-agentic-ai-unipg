# SEMANTIC CHUNKING - parsing.py

from bs4 import BeautifulSoup
from datetime import datetime, timezone
from pathlib import Path
import logging
import warnings

from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document

from scraping import sha

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="camelot")


def extract_units_from_elements(elements, source_url, page_title, doc_type, file_name=None):
    """
    Converte gli elementi Unstructured in 'unità' piccole e ordinate (paragrafi, list item, tabelle).
    NON fa chunking semantico qui: restituisce unità atomiche su cui l'indexing farà semantic grouping.
    """
    from unstructured.documents.elements import Title, Header, NarrativeText, ListItem, Text, Table

    crawl_ts = datetime.now(timezone.utc).isoformat()
    units: list[Document] = []
    current_header: str | None = None
    unit_idx = 0

    for el in elements:
        text = getattr(el, "text", None)
        if not text:
            continue
        text = text.strip()
        if not text:
            continue

        # Catturo "sezioni" come contesto, ma non le tratto come chunk intero:
        if isinstance(el, (Title, Header)):
            current_header = text
            continue

        # Testo "normale" -> unità atomica
        if isinstance(el, (NarrativeText, ListItem, Text)):
            # Mantieni un minimo di contesto (header) come prefisso del contenuto dell'unità
            content = f"{current_header}\n{text}" if current_header else text

            units.append(Document(
                page_content=content,
                metadata={
                    "source_url": source_url,
                    "doc_type": doc_type,
                    "page_title": page_title,
                    "file_name": file_name,
                    "element_type": "SemanticUnit",
                    "lang": ["ita"],
                    "crawl_ts": crawl_ts,
                    "unit_id": sha(f"{source_url}_unit_{unit_idx}"),
                },
            ))
            unit_idx += 1
            continue

        # Tabelle -> unità atomica dedicata
        if isinstance(el, Table):
            table_txt = f"[TABELLA]\n{text}"
            if current_header:
                table_txt = f"{current_header}\n{table_txt}"

            units.append(Document(
                page_content=table_txt,
                metadata={
                    "source_url": source_url,
                    "doc_type": doc_type,
                    "page_title": page_title,
                    "file_name": file_name,
                    "element_type": "SemanticUnitTable",
                    "lang": ["ita"],
                    "crawl_ts": crawl_ts,
                    "unit_id": sha(f"{source_url}_table_{unit_idx}"),
                },
            ))
            unit_idx += 1
            continue

    print(f"[INFO] Estratte {len(units)} unità atomiche da {source_url}")
    return units


def to_documents_from_html(file_path: Path, source_url: str, page_title: str) -> list[Document]:
    raw_html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")

    main_el = soup.find("main")
    if not main_el:
        print(f"[WARN] Nessun <main> trovato in {source_url}, salto")
        return []

    # Rimuovi moduli inutili come nel tuo codice section-based
    for mod in main_el.select("div.module-container.col-xs-12"):
        mod.decompose()

    main_html = str(main_el)

    elements = partition_html(
        text=main_html,
        include_page_breaks=False,
        languages=["ita", "eng"]
    )

    # fallback se unstructured non estrae nulla
    if not elements:
        text_fallback = main_el.get_text(separator="\n", strip=True)
        if not text_fallback:
            return []
        return [Document(
            page_content=text_fallback,
            metadata={
                "source_url": source_url,
                "doc_type": "html",
                "page_title": page_title,
                "file_name": file_path.name,
                "element_type": "FallbackText",
                "lang": ["ita"],
                "crawl_ts": datetime.now(timezone.utc).isoformat(),
                "unit_id": sha(source_url),
            },
        )]

    return extract_units_from_elements(elements, source_url, page_title, doc_type="html", file_name=file_path.name)


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

    return extract_units_from_elements(elements, source_url, page_title=file_path.stem, doc_type="pdf", file_name=file_path.name)
