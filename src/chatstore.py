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


def _scope(user: str, paper: str | None = None) -> dict:
    """Chroma `where` filter scoped to one user (+ optional paper session)."""
    clauses = [{"user": user or "\x00"}]
    if paper is not None:
        clauses.append({"paper": paper})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def save_exchange(user: str, paper: str, query: str, answer: str, report, chunks) -> None:
    """Store one Q&A exchange for `user` under `paper` ('*' = all-papers session)."""
    payload = {"query": query, "answer": answer, "report": report, "chunks": chunks}
    _get().add(
        ids=[uuid.uuid4().hex],
        documents=[query or " "],
        metadatas=[{"user": user, "paper": paper, "ts": time.time(), "payload": json.dumps(payload)}],
    )


def list_chats(user: str, paper: str):
    """Return `user`'s stored exchanges for `paper`, oldest first."""
    data = _get().get(where=_scope(user, paper), include=["metadatas"])
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


def counts(user: str) -> dict:
    """Return {paper_key: exchange_count} across `user`'s sessions."""
    data = _get().get(where=_scope(user), include=["metadatas"])
    out = {}
    for m in (data.get("metadatas") or []):
        p = m.get("paper", "*")
        out[p] = out.get(p, 0) + 1
    return out


def clear(user: str, paper: str | None = None) -> None:
    """Delete one of `user`'s sessions, or all of the user's chats if `paper` is None.

    Scoped to the user - never drops the shared collection.
    """
    _get().delete(where=_scope(user, paper))
