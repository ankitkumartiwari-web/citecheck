"""ChromaDB vector store.

ChromaDB ships with a built-in embedding model (a small ONNX MiniLM) that runs
locally on your CPU. That means embeddings are FREE and need no API key — only
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
        # Default local embedding function — no API, no PyTorch.
        embed_fn = embedding_functions.DefaultEmbeddingFunction()
        _collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def stats():
    """Quick info for the UI: how many chunks and which papers are indexed."""
    col = get_collection()
    data = col.get(include=["metadatas"])
    metas = data.get("metadatas") or []
    sources = sorted({m.get("source", "unknown") for m in metas})
    return {"chunks": len(metas), "papers": sources}


def paper_context(source: str, limit: int = 14, max_chars: int = 9000) -> str:
    """Return a representative slice of one paper's text (first `limit` chunks)."""
    col = get_collection()
    data = col.get(where={"source": source}, include=["documents"], limit=limit)
    docs = data.get("documents") or []
    text = "\n\n".join(docs)
    return text[:max_chars]


def query_in_paper(source: str, query: str, k: int = 3):
    """Top-k passages for `query` restricted to a single paper."""
    col = get_collection()
    res = col.query(query_texts=[query], n_results=k, where={"source": source})
    docs = (res.get("documents") or [[]])[0]
    return [d for d in docs if d]


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
