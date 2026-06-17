# CiteCheck: Research Paper Assistant with Citation Verification

A full-stack **RAG (Retrieval-Augmented Generation)** application for academic
papers. Ask natural-language questions and get answers that are **grounded in the
source text** and **fact-checked claim by claim**, so hallucinations are flagged
instead of trusted. On top of grounded chat it adds cross-paper comparison, a
multi-agent peer-review panel, translation, a trained ML classifier, and
per-account data isolation behind a login.

Built to run for **free**: embeddings run locally (ChromaDB), and answering uses
**OpenRouter's free model tier** (OpenAI-compatible).

Live demo: https://citecheck.up.railway.app/

---

## Why it stands out

Most "chat with your PDF" demos stop at retrieve-then-answer. CiteCheck adds the
parts that separate a real RAG engineer from a tutorial:

- **A verification layer.** After the model answers, a second pass splits the
  answer into atomic claims and labels each one **supported / partially supported
  / unsupported** against the retrieved passages, with the supporting citation
  ids. Ungrounded statements are surfaced, not hidden.
- **A trained ML model.** A scikit-learn classifier tags each retrieved passage
  with its rhetorical role (Background / Method / Result / Conclusion), a real
  trained model, not just API calls.
- **Multi-agent orchestration.** The peer-review panel runs three specialist
  reviewer agents plus a chair that synthesizes a recommendation.
- **Real product concerns.** Accounts with strict per-user data isolation,
  rate limiting to protect free credits, load balancing across multiple API keys,
  and a flat, responsive UI.

---

## Features

| Mode | What it does |
|------|--------------|
| **Chat** | Grounded answers with inline citations, claim-by-claim verification, an interactive evidence graph (claims to source passages), and ML role tags on each passage. Chats are scoped per selected paper and persisted. |
| **Compare** | Cross-paper consensus: pick two or more papers and a claim, and see where they **support**, **refute**, or stay **neutral**, with evidence quotes. |
| **Peer Review** | A multi-agent panel: three specialist reviewers (methodology, novelty, clarity) plus a chair that synthesizes a decision and score. |
| **Translate** | Translate an indexed paper's text into another language in a single call. |

Plus: **accounts** (each user's papers, chats, and stats are private), **PDF
export** on every result, and a separate marketing **landing page**.

---

## How it works

```
   PDFs ──chunk──▶ embed (local ONNX, free) ──▶ ChromaDB (per user)
                                                     │
                                                     │ top-K passages
   Question ─────────────────────────────────────────▶ LLM answers, cites [1][2]
                                                     │
                                                     ▼
                       LLM fact-checks each claim vs. the passages
                       (supported / partial / unsupported)
                                                     │
                                                     ▼
                       trained ML tags passage roles + evidence graph
```

| Stage | Tech | Cost |
|-------|------|------|
| PDF parsing | `pypdf` | free |
| Chunking | `langchain-text-splitters` | free |
| Embeddings | ChromaDB built-in ONNX model (local) | free |
| Vector store | ChromaDB (persisted to disk) | free |
| Answering / verification | OpenRouter free model, OpenAI-compatible | free |
| Passage role tagging | scikit-learn TF-IDF + LogisticRegression | free |
| Accounts | SQLite + PBKDF2 password hashing (stdlib) | free |
| Rate limiting | local throttle + daily cap + multi-key load balancing | protects credits |
| Backend | FastAPI (serves API + frontend) | free |
| Frontend | vanilla JS, flat editorial UI | free |

---

## Setup

> **Python note:** use **Python 3.11 or 3.12** for the smoothest install. Very new
> versions sometimes lack prebuilt wheels for the ML packages.

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your free OpenRouter key(s)
copy .env.example .env          # Windows  (cp on macOS/Linux)
#   then edit .env and paste a key from https://openrouter.ai/keys
```

Add a second key for higher throughput. Calls are round-robined across all keys:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-first-key
OPENROUTER_API_KEY_2=sk-or-v1-your-second-key
```

---

## Run it

```bash
# (optional) grab a sample paper to test with
python scripts/download_sample.py

# launch the web app (FastAPI backend + responsive frontend)
python server.py
#   then open http://127.0.0.1:8000
```

On first launch you create an account, then upload PDFs from the sidebar and start
asking. The four modes (Chat, Compare, Peer Review, Translate) share a flat,
responsive UI (the sidebar collapses on mobile).

A command-line interface is also available for quick testing (it uses a single
local library, no login):

```bash
python main.py ingest                                  # index data/papers/*.pdf
python main.py ask "What problem does this paper solve?"
```

---

## Accounts and privacy

Every data endpoint requires a logged-in user. Isolation is enforced at the data
layer, not just the UI:

- Passwords are hashed with **PBKDF2-HMAC-SHA256**; sessions are signed cookies,
  both stored in a local SQLite database.
- Every vector-store and chat read is scoped by username with a filter that
  **fails closed** (a missing user matches nothing), so no account can ever see
  another account's papers, chunks, chats, or stats.
- Uploaded PDFs are stored in per-user folders.

---

## Trained ML model

`ml/` contains a small, honestly-evaluated classifier (not an API wrapper):

- **Features:** TF-IDF over word (1–2) and character (3–5) n-grams.
- **Model:** Logistic Regression predicting a sentence's rhetorical role:
  Background, Method, Result, or Conclusion.
- **Evaluation:** stratified 5-fold cross-validation.
- **Use:** retrieved passages are tagged with their predicted role in the Chat view.

```bash
python -m ml.train      # retrain and persist ml/model.joblib
```

---

## Rate limiting (protects your free credits)

The app caps LLM usage so it cannot drain your free OpenRouter quota:

- **Minimum interval between calls** (stays under OpenRouter's ~20 requests/min
  free limit); the interval shrinks as you add more keys.
- **Daily cap** that scales with the number of keys (~40 calls/key/day by default).
- **Automatic exponential backoff** on 429 / network errors, with failover across
  keys.

Tune in `.env`: `MAX_CALLS_PER_DAY`, `MIN_SECONDS_BETWEEN_CALLS`, `MAX_RETRIES`.
The sidebar shows a live "calls today" meter. (OpenRouter gives 50 free
requests/day per key, or 1000/day per key once you've purchased >= $10 of credit.)

---

## Project structure

```
.
├── server.py               # FastAPI backend, auth, and serves the frontend
├── main.py                 # command-line interface
├── config.py               # all settings in one place
├── requirements.txt
├── .env.example            # copy to .env and add your key(s)
├── src/
│   ├── auth.py             # accounts + sessions (SQLite, PBKDF2)
│   ├── ingest.py           # PDF -> chunks -> vector store (per user)
│   ├── vectorstore.py      # ChromaDB + local embeddings, user-scoped reads
│   ├── retriever.py        # similarity search (+ optional re-ranking)
│   ├── rag.py              # ties retrieve -> answer -> verify together
│   ├── llm.py              # OpenRouter client, multi-key load balancing
│   ├── verifier.py         # claim-by-claim grounding check
│   ├── compare.py          # cross-paper consensus / contradiction
│   ├── review.py           # multi-agent peer-review panel
│   ├── translate.py        # paper translation
│   ├── chatstore.py        # per-user chat history in ChromaDB
│   └── ratelimit.py        # throttle + daily cap
├── ml/
│   ├── dataset.py          # training data
│   ├── train.py            # train + cross-validate the role classifier
│   ├── classify.py         # inference helper
│   └── model.joblib        # persisted trained model
├── frontend/               # the app UI (chat, compare, review, translate, auth)
├── landing/                # standalone marketing landing page
├── scripts/
│   └── download_sample.py  # fetch a sample arXiv paper
└── data/
    ├── papers/             # uploaded PDFs (per-user subfolders)
    ├── chroma/             # vector DB (auto-created)
    └── auth.db             # accounts + sessions (auto-created)
```

---

## Deploy

The app is one FastAPI web service: `server.py` serves both the API and
`frontend/`. The marketing page in `landing/` is a static site that can be hosted
separately.

### App (Railway)

`railway.json` and `Procfile` are included. Railway runs `python server.py`. Set
the environment variables in the Railway dashboard:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-real-key
# optional, for higher throughput:
OPENROUTER_API_KEY_2=sk-or-v1-your-second-key
# optional, persistent storage for uploads + vector DB across redeploys:
DATA_DIR=/data
```

### App (any Python host)

Use these settings on any host that provides a `PORT` variable (a `render.yaml`
is also included for Render):

```bash
Build command: pip install -r requirements.txt
Start command: uvicorn server:app --host 0.0.0.0 --port $PORT
```

Set `DATA_DIR` to a persistent path so uploaded PDFs and the vector database
survive restarts.

### Landing page (Netlify)

`netlify.toml` publishes the `landing/` folder as a static site with sensible
security headers.

Do not commit `.env`, uploaded PDFs, `data/chroma/`, or `data/auth.db`; they are
intentionally ignored.

---

## Talking points

- **RAG pipeline:** chunking strategy, local embeddings, vector similarity search.
- **Grounding / hallucination detection:** the verification pass and why naive
  RAG is risky without it.
- **Structured outputs:** forcing the LLM into a typed schema (Pydantic) for
  reliable, parseable verification and review results.
- **Multi-tenancy:** per-user data isolation enforced at the query layer.
- **Cost engineering:** local embeddings plus a free LLM tier, with rate limiting
  and multi-key load balancing.

## Ideas to extend

- Add **cross-encoder re-ranking** (`sentence-transformers`) and measure the
  retrieval-quality improvement.
- Add an **evaluation script** for answer faithfulness on a small test set.
- Support **arXiv search/import by ID** instead of manual upload.
- Highlight the **exact supporting sentence** inside each passage for a claim.
