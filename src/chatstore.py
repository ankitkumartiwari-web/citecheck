"""Persist chat exchanges in ChromaDB (a separate `chat_history` collection).

Each exchange is stored with the question as the embedded document (so past
questions are semantically searchable) and the full payload (answer, report,
sources) plus the paper key and timestamp in metadata.
"""
import json
import time
import uuid

import chromadb
from chromadb.utils import embedding_functions

import config

_COLLECTION = "chat_history"
_collection = None


def _get():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        ef = embedding_functions.DefaultEmbeddingFunction()
        _collection = client.get_or_create_collection(name=_COLLECTION, embedding_function=ef)
    return _collection


def save_exchange(paper: str, query: str, answer: str, report, chunks) -> None:
    """Store one Q&A exchange under `paper` (use '*' for the all-papers session)."""
    payload = {"query": query, "answer": answer, "report": report, "chunks": chunks}
    _get().add(
        ids=[uuid.uuid4().hex],
        documents=[query or " "],
        metadatas=[{"paper": paper, "ts": time.time(), "payload": json.dumps(payload)}],
    )


def list_chats(paper: str):
    """Return stored exchanges for `paper`, oldest first."""
    data = _get().get(where={"paper": paper}, include=["metadatas"])
    items = []
    for m in (data.get("metadatas") or []):
        try:
            it = json.loads(m["payload"])
            it["_ts"] = m.get("ts", 0)
            items.append(it)
        except Exception:
            pass
    items.sort(key=lambda x: x.get("_ts", 0))
    for it in items:
        it.pop("_ts", None)
    return items


def counts() -> dict:
    """Return {paper_key: exchange_count} across all sessions."""
    data = _get().get(include=["metadatas"])
    out = {}
    for m in (data.get("metadatas") or []):
        p = m.get("paper", "*")
        out[p] = out.get(p, 0) + 1
    return out


def clear(paper: str | None = None) -> None:
    """Delete one session's chats, or all if `paper` is None."""
    global _collection
    if paper:
        _get().delete(where={"paper": paper})
    else:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        try:
            client.delete_collection(_COLLECTION)
        except Exception:
            pass
        _collection = None
