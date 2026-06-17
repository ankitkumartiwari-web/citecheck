"""CiteCheck backend - FastAPI.

Serves the custom frontend (frontend/) and a small JSON API that wraps the same
RAG pipeline used everywhere else (retrieve -> answer -> verify), with the
rate limiting from src/ratelimit.py.

Accounts: every request to a data endpoint requires a logged-in user (a signed
session cookie). All papers, chats and stats are isolated per account, so no
user can ever see another user's uploads.

Run from the project root:
    python server.py
    # or: uvicorn server:app --reload
Then open http://127.0.0.1:8000
"""
from pathlib import Path
from typing import List
from urllib.parse import unquote

from fastapi import Cookie, Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from src import auth, chatstore
from src.compare import compare_claim
from src.ingest import ingest_dir, ingest_pdf
from src.rag import ask
from src.ratelimit import DailyLimitReached, usage_today
from src.review import review_paper
from src.translate import translate_paper
from src.vectorstore import clear_user, delete_paper, stats

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 days

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


# ───────────────────────── Auth ─────────────────────────
def require_user(cc_session: str | None = Cookie(default=None)) -> str:
    """Resolve the session cookie to a username, or 401 if not logged in."""
    user = auth.username_for_token(cc_session)
    if not user:
        raise HTTPException(401, "Not authenticated.")
    return user


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=config.SESSION_COOKIE, value=token, httponly=True,
        samesite="lax", max_age=SESSION_MAX_AGE, path="/",
    )


def _user_papers_dir(user: str) -> Path:
    """Per-user folder for uploaded PDFs (isolated from other accounts)."""
    d = config.PAPERS_DIR / user
    d.mkdir(parents=True, exist_ok=True)
    return d


class AuthBody(BaseModel):
    username: str
    password: str


@app.post("/api/register")
def post_register(body: AuthBody, response: Response):
    try:
        username = auth.register(body.username, body.password)
    except auth.AuthError as e:
        raise HTTPException(400, str(e))
    _set_session_cookie(response, auth.create_session(username))
    return {"username": username}


@app.post("/api/login")
def post_login(body: AuthBody, response: Response):
    try:
        username = auth.verify(body.username, body.password)
    except auth.AuthError as e:
        raise HTTPException(401, str(e))
    _set_session_cookie(response, auth.create_session(username))
    return {"username": username}


@app.post("/api/logout")
def post_logout(response: Response, cc_session: str | None = Cookie(default=None)):
    auth.delete_session(cc_session)
    response.delete_cookie(config.SESSION_COOKIE, path="/")
    return {"ok": True}


@app.get("/api/me")
def get_me(user: str = Depends(require_user)):
    return {"username": user}


# ───────────────────────── Request models ─────────────────────────
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


def _stats_payload(user: str) -> dict:
    s = stats(user)
    used, cap = usage_today()  # the API-key budget is shared (global), not per-user
    try:
        chat_counts = chatstore.counts(user)
    except Exception:
        chat_counts = {}
    return {
        "chunks": s["chunks"],
        "papers": s["papers"],
        "calls_used": used,
        "calls_cap": cap,
        "chat_counts": chat_counts,
    }


# ───────────────────────── Data endpoints (all require a user) ─────────────────────────
@app.get("/api/stats")
def get_stats(user: str = Depends(require_user)):
    return _stats_payload(user)


@app.post("/api/ask")
def post_ask(req: AskRequest, user: str = Depends(require_user)):
    """Run the full RAG pipeline. Defined as a sync function so the blocking
    network calls run in FastAPI's threadpool."""
    if not req.query.strip():
        raise HTTPException(400, "Empty query.")
    if stats(user)["chunks"] == 0:
        raise HTTPException(400, "No documents indexed yet. Upload a PDF first.")
    try:
        result = ask(req.query, do_verify=req.verify, sources=req.sources, user=user)
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
            chatstore.save_exchange(user, paper_key, req.query, result["answer"], report_d, result["chunks"])
        except Exception:
            pass

    return {"answer": result["answer"], "chunks": result["chunks"], "report": report_d}


@app.post("/api/compare")
def post_compare(req: CompareRequest, user: str = Depends(require_user)):
    if not req.claim.strip():
        raise HTTPException(400, "Empty claim.")
    try:
        result = compare_claim(req.claim, papers=req.papers, user=user)
    except DailyLimitReached as e:
        raise HTTPException(429, str(e))
    except Exception as e:
        raise HTTPException(502, f"LLM request failed: {e}")
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return result


@app.post("/api/review")
def post_review(req: ReviewRequest, user: str = Depends(require_user)):
    if not req.paper.strip():
        raise HTTPException(400, "No paper selected.")
    try:
        result = review_paper(req.paper, user)
    except DailyLimitReached as e:
        raise HTTPException(429, str(e))
    except Exception as e:
        raise HTTPException(502, f"LLM request failed: {e}")
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return result


@app.get("/api/chats")
def get_chats(paper: str = "*", user: str = Depends(require_user)):
    return {"paper": paper, "items": chatstore.list_chats(user, paper)}


@app.post("/api/chats/clear")
def post_chats_clear(req: ChatClearRequest, user: str = Depends(require_user)):
    chatstore.clear(user, req.paper)
    return {"ok": True, **_stats_payload(user)}


@app.post("/api/translate")
def post_translate(req: TranslateRequest, user: str = Depends(require_user)):
    if not req.paper.strip() or not req.language.strip():
        raise HTTPException(400, "Paper and language are required.")
    try:
        result = translate_paper(req.paper, req.language, user)
    except DailyLimitReached as e:
        raise HTTPException(429, str(e))
    except Exception as e:
        raise HTTPException(502, f"LLM request failed: {e}")
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return result


@app.post("/api/ingest")
def post_ingest(files: List[UploadFile] = File(...), user: str = Depends(require_user)):
    results = {}
    dest_dir = _user_papers_dir(user)
    for f in files:
        name = Path(f.filename or "upload.pdf").name
        dest = dest_dir / name
        dest.write_bytes(f.file.read())
        results[name] = ingest_pdf(dest, user)
    return {"results": results, **_stats_payload(user)}


@app.post("/api/ingest-folder")
def post_ingest_folder(user: str = Depends(require_user)):
    """Ingest the bundled sample PDFs in data/papers/ into this user's library."""
    total, results = ingest_dir(user)
    return {"results": results, "total": total, **_stats_payload(user)}


@app.delete("/api/papers/{paper:path}")
def delete_paper_route(paper: str, user: str = Depends(require_user)):
    paper = Path(unquote(paper)).name.strip()
    if not paper:
        raise HTTPException(400, "No paper selected.")

    file_path = _user_papers_dir(user) / paper
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            raise HTTPException(500, f"Failed to delete PDF file: {e}")

    deleted_chunks = delete_paper(paper, user)
    try:
        chatstore.clear(user, paper)
    except Exception:
        pass

    return {"ok": True, "paper": paper, "deleted_chunks": deleted_chunks, **_stats_payload(user)}


@app.post("/api/clear")
def post_clear(user: str = Depends(require_user)):
    """Clear only THIS user's library - papers, chunks, files and chats."""
    clear_user(user)
    try:
        chatstore.clear(user)
    except Exception:
        pass
    user_dir = _user_papers_dir(user)
    for pdf in user_dir.glob("*.pdf"):
        try:
            pdf.unlink()
        except Exception:
            pass
    return {"ok": True, **_stats_payload(user)}


@app.get("/healthz")
def healthz():
    """Liveness probe for the host (Railway health check). No auth."""
    return {"ok": True}


# Serve the frontend LAST so the /api/* routes above take precedence.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import os

    import uvicorn

    # Bind to 0.0.0.0 and the host-provided $PORT so the platform proxy can reach
    # us in a container (localhost would only be reachable inside the container).
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
