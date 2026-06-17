"""ChromaDB vector store.

ChromaDB ships with a built-in embedding model (a small ONNX MiniLM) that runs
locally on your CPU. That means embeddings are FREE and need no API key - only
a one-time ~80MB model download the first time you ingest a document.
"""
import chromadb
from chromadb.utils import embedding_functions

import config

_collection = None


def get_collection():
    """Return the (singleton) persistent ChromaDB collection."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        # Default local embedding function - no API, no PyTorch.
        embed_fn = embedding_functions.DefaultEmbeddingFunction()
        _collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def where_for(user: str, source: str | None = None, sources=None) -> dict:
    """Build a Chroma `where` filter scoped to one user (+ optional paper(s)).

    Every read MUST go through this so one account can never see another's data.
    A falsy `user` yields a filter that matches nothing (fail closed).
    """
    clauses = [{"user": user or "\x00"}]
    if source is not None:
        clauses.append({"source": source})
    elif sources:
        clauses.append({"source": {"$in": list(sources)}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def stats(user: str):
    """Quick info for the UI: chunks and papers indexed FOR THIS USER."""
    col = get_collection()
    data = col.get(where=where_for(user), include=["metadatas"])
    metas = data.get("metadatas") or []
    sources = sorted({m.get("source", "unknown") for m in metas})
    return {"chunks": len(metas), "papers": sources}


def paper_context(source: str, user: str, limit: int = 14, max_chars: int = 9000) -> str:
    """Return a representative slice of one of the user's papers (first `limit` chunks)."""
    col = get_collection()
    data = col.get(where=where_for(user, source), include=["documents"], limit=limit)
    docs = data.get("documents") or []
    text = "\n\n".join(docs)
    return text[:max_chars]


def query_in_paper(source: str, query: str, user: str, k: int = 3):
    """Top-k passages for `query` restricted to a single paper owned by `user`."""
    col = get_collection()
    res = col.query(query_texts=[query], n_results=k, where=where_for(user, source))
    docs = (res.get("documents") or [[]])[0]
    return [d for d in docs if d]


def delete_paper(source: str, user: str) -> int:
    """Delete one of the user's papers (all its chunks). Returns the number removed."""
    col = get_collection()
    data = col.get(where=where_for(user, source), include=[])
    ids = data.get("ids") or []
    if ids:
        col.delete(ids=ids)
    return len(ids)


def clear_user(user: str) -> int:
    """Delete every chunk owned by `user` (their whole library). Returns the count."""
    col = get_collection()
    data = col.get(where=where_for(user), include=[])
    ids = data.get("ids") or []
    if ids:
        col.delete(ids=ids)
    return len(ids)


def reset():
    """Delete everything in the collection (useful for a clean re-index)."""
    global _collection
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    _collection = None
    return get_collection()
