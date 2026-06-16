"""Orchestration: tie retrieval, answering, and verification together."""
from ml.classify import classify_passage
from src.llm import answer
from src.retriever import retrieve
from src.verifier import verify


def _tag_roles(chunks):
    """Annotate each chunk with the trained model's predicted rhetorical role."""
    for c in chunks:
        c["role"] = classify_passage(c["text"])  # None if model unavailable
    return chunks


def ask(query: str, k: int | None = None, do_verify: bool = True, sources=None) -> dict:
    """Full RAG pipeline for one question.

    Returns:
        {
          "answer": str,
          "chunks": list of retrieved passages,
          "report": VerificationReport | None,
        }
    """
    chunks = retrieve(query, k=k, sources=sources)
    if not chunks:
        msg = ("No relevant passages found in the selected paper(s)."
               if sources else
               "No documents are indexed yet. Add PDFs and ingest them first.")
        return {"answer": msg, "chunks": [], "report": None}

    _tag_roles(chunks)
    ans = answer(query, chunks)
    report = verify(ans, chunks) if do_verify else None
    return {"answer": ans, "chunks": chunks, "report": report}
