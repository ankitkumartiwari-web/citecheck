"""Retrieval: turn a question into the most relevant passages.

The base path uses ChromaDB's vector similarity search (free, local).
An OPTIONAL cross-encoder re-ranker can sharpen the ordering if you install
`sentence-transformers` (see requirements.txt).
"""
import config
from src.vectorstore import get_collection, where_for

# Optional re-ranker - only loaded if the package is installed.
_reranker = None
try:
    from sentence_transformers import CrossEncoder  # type: ignore

    _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    _reranker = None  # base vector search still works fine without it


def retrieve(query: str, k: int | None = None, sources=None, user: str | None = None):
    """Return a list of passage dicts ranked by relevance to `query`.

    Each passage: {id, text, source, page, distance}
    `id` is a 1-based number we use for inline citations like [1], [2].
    Retrieval is always restricted to `user`'s own papers; if `sources` is a
    non-empty list of filenames, it is further scoped to those papers.
    """
    k = k or config.TOP_K
    collection = get_collection()

    # Always scope to the owner; optionally narrow to the selected papers.
    where = where_for(user, sources=sources)

    # Pull a few extra candidates when re-ranking so it has room to reorder.
    n_fetch = k * 3 if _reranker else k
    res = collection.query(query_texts=[query], n_results=n_fetch, where=where)

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    candidates = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        candidates.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page"),
            "distance": dists[i] if i < len(dists) else None,
        })

    if _reranker and candidates:
        pairs = [(query, c["text"]) for c in candidates]
        scores = _reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    top = candidates[:k]
    for i, c in enumerate(top):
        c["id"] = i + 1  # 1-based citation id
    return top
