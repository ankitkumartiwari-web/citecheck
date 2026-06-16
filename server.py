"""CiteCheck backend — FastAPI.

Serves the custom frontend (frontend/) and a small JSON API that wraps the same
RAG pipeline used everywhere else (retrieve -> answer -> verify), with the
rate limiting from src/ratelimit.py.

Run from the project root:
    python server.py
    # or: uvicorn server:app --reload
Then open http://127.0.0.1:8000
"""
from pathlib import Path
import os
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from src import chatstore
from src.compare import compare_claim
from src.ingest import ingest_dir, ingest_pdf
from src.rag import ask
from src.ratelimit import DailyLimitReached, usage_today
from src.review import review_paper
from src.translate import translate_paper
from src.vectorstore import reset, stats

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="CiteCheck")


@app.middleware("http")
async def no_cache_static(request, call_next):
    """Stop the browser from serving stale app.js/styles.css/index.html."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


class AskRequest(BaseModel):
    query: str
    verify: bool = True
    sources: list[str] | None = None   # scope chat to selected papers


class CompareRequest(BaseModel):
    claim: str
    papers: list[str] | None = None


class ReviewRequest(BaseModel):
    paper: str


class TranslateRequest(BaseModel):
    paper: str
    language: str


class ChatClearRequest(BaseModel):
    paper: str | None = None


def _stats_payload() -> dict:
    s = stats()
    used, cap = usage_today()
    try:
        chat_counts = chatstore.counts()
    except Exception:
        chat_counts = {}
    return {
        "chunks": s["chunks"],
        "papers": s["papers"],
        "calls_used": used,
        "calls_cap": cap,
        "chat_counts": chat_counts,
    }


@app.get("/api/stats")
def get_stats():
    return _stats_payload()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/ask")
def post_ask(req: AskRequest):
    """Run the full RAG pipeline. Defined as a sync function so the blocking
    network calls run in FastAPI's threadpool."""
    if not req.query.strip():
        raise HTTPException(400, "Empty query.")
    if stats()["chunks"] == 0:
        raise HTTPException(400, "No documents indexed yet. Upload a PDF first.")
    try:
        result = ask(req.query, do_verify=req.verify, sources=req.sources)
    except DailyLimitReached as e:
        raise HTTPException(429, str(e))
    except Exception as e:  # surface a clean message to the UI
        raise HTTPException(502, f"LLM request failed: {e}")

    report = result.get("report")
    report_d = report.model_dump() if report is not None else None

    # Persist the exchange in the vector DB (skip the "no passages" non-answers).
    if result.get("chunks"):
        paper_key = req.sources[0] if (req.sources and len(req.sources) == 1) else "*"
        try:
            chatstore.save_exchange(paper_key, req.query, result["answer"], report_d, result["chunks"])
        except Exception:
            pass

    return {"answer": result["answer"], "chunks": result["chunks"], "report": report_d}


@app.post("/api/compare")
def post_compare(req: CompareRequest):
    if not req.claim.strip():
        raise HTTPException(400, "Empty claim.")
    try:
        result = compare_claim(req.claim, papers=req.papers)
    except DailyLimitReached as e:
        raise HTTPException(429, str(e))
    except Exception as e:
        raise HTTPException(502, f"LLM request failed: {e}")
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return result


@app.post("/api/review")
def post_review(req: ReviewRequest):
    if not req.paper.strip():
        raise HTTPException(400, "No paper selected.")
    try:
        result = review_paper(req.paper)
    except DailyLimitReached as e:
        raise HTTPException(429, str(e))
    except Exception as e:
        raise HTTPException(502, f"LLM request failed: {e}")
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return result


@app.get("/api/chats")
def get_chats(paper: str = "*"):
    return {"paper": paper, "items": chatstore.list_chats(paper)}


@app.post("/api/chats/clear")
def post_chats_clear(req: ChatClearRequest):
    chatstore.clear(req.paper)
    return {"ok": True, **_stats_payload()}


@app.post("/api/translate")
def post_translate(req: TranslateRequest):
    if not req.paper.strip() or not req.language.strip():
        raise HTTPException(400, "Paper and language are required.")
    try:
        result = translate_paper(req.paper, req.language)
    except DailyLimitReached as e:
        raise HTTPException(429, str(e))
    except Exception as e:
        raise HTTPException(502, f"LLM request failed: {e}")
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return result


@app.post("/api/ingest")
def post_ingest(files: List[UploadFile] = File(...)):
    results = {}
    for f in files:
        name = Path(f.filename or "upload.pdf").name
        dest = config.PAPERS_DIR / name
        dest.write_bytes(f.file.read())
        results[name] = ingest_pdf(dest)
    return {"results": results, **_stats_payload()}


@app.post("/api/ingest-folder")
def post_ingest_folder():
    total, results = ingest_dir()
    return {"results": results, "total": total, **_stats_payload()}


@app.post("/api/clear")
def post_clear():
    reset()
    return {"ok": True, **_stats_payload()}


# Serve the frontend LAST so the /api/* routes above take precedence.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, reload=False)
