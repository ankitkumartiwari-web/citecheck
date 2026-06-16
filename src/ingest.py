"""Ingestion pipeline: PDF -> text -> chunks -> vector store.

Run from the project root:
    python -m src.ingest            # ingest everything in data/papers/
    python -m src.ingest paper.pdf  # ingest a single file
"""
import hashlib
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config
from src.vectorstore import get_collection


def _read_pdf_pages(path: Path):
    """Yield (page_number, text) for each non-empty page of a PDF."""
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            yield i + 1, text


def ingest_pdf(path) -> int:
    """Chunk one PDF and add its chunks to the vector store. Returns chunk count."""
    path = Path(path)
    collection = get_collection()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    docs, metas, ids = [], [], []
    for page_num, page_text in _read_pdf_pages(path):
        for chunk in splitter.split_text(page_text):
            chunk = chunk.strip()
            if not chunk:
                continue
            # Stable id so re-ingesting the same paper updates instead of duplicating.
            uid = hashlib.sha1(
                f"{path.name}-{page_num}-{chunk[:80]}".encode("utf-8")
            ).hexdigest()
            docs.append(chunk)
            metas.append({"source": path.name, "page": page_num})
            ids.append(uid)

    if docs:
        collection.upsert(documents=docs, metadatas=metas, ids=ids)
    return len(docs)


def ingest_dir(directory=None):
    """Ingest every PDF in a directory. Returns (total_chunks, {filename: count})."""
    directory = Path(directory or config.PAPERS_DIR)
    results, total = {}, 0
    for pdf in sorted(directory.glob("*.pdf")):
        n = ingest_pdf(pdf)
        results[pdf.name] = n
        total += n
    return total, results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        count = ingest_pdf(sys.argv[1])
        print(f"Ingested {count} chunks from {sys.argv[1]}")
    else:
        total, results = ingest_dir()
        for name, n in results.items():
            print(f"  {name}: {n} chunks")
        print(f"Done. {total} chunks total from {len(results)} paper(s).")
