# 📚 CiteCheck — Research Paper Assistant with Citation Verification

A full-stack **RAG (Retrieval-Augmented Generation)** project: ask natural-language
questions about academic papers and get answers that are **grounded in the source
text** and **fact-checked claim by claim** — so hallucinations get flagged instead
of trusted.

Built to be **100% free to run**: embeddings run locally (ChromaDB), and the LLM
uses **Google Gemini's free tier**.

---

## ✨ What makes this project stand out

Most "chat with your PDF" demos stop at retrieval + answer. This one adds a
**verification layer**: after the model answers, a second pass breaks the answer
into atomic claims and labels each as **supported / partially supported /
unsupported** against the retrieved passages. That grounding check is exactly the
kind of thing that separates a real RAG engineer from a tutorial-follower.

---

## 🏗️ How it works

```
                        ┌─────────────────────────────┐
  PDFs ── chunk ──▶ embed (local, free) ──▶ ChromaDB   │
                        └──────────────┬──────────────┘
                                       │  top-K passages
   Question ───────────────────────────▶  Gemini answers, cites [1][2]
                                       │
                                       ▼
                 Gemini fact-checks each claim vs. the passages
                 (supported / partial / unsupported)
```

| Stage         | Tech                                   | Cost |
|---------------|----------------------------------------|------|
| PDF parsing   | `pypdf`                                | free |
| Chunking      | `langchain-text-splitters`             | free |
| Embeddings    | ChromaDB built-in ONNX model (local)   | free |
| Vector store  | ChromaDB (persisted to disk)           | free |
| Answering     | OpenRouter free model (OpenAI-compatible) | free |
| Verification  | OpenRouter free model (JSON output)    | free |
| Rate limiting | local throttle + daily cap             | protects credits |
| UI            | Custom FastAPI backend + JS frontend   | free |

---

## 🚀 Setup

> **Python note:** use **Python 3.11 or 3.12** for the smoothest install. Very new
> versions (3.14) sometimes lack prebuilt wheels for ML packages.

```bash
# 1. (Recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your free OpenRouter key
copy .env.example .env         # Windows  (cp on macOS/Linux)
#   then edit .env and paste your key from https://openrouter.ai/keys
```

---

## ▶️ Run it

```bash
# (optional) grab a sample paper to test with
python scripts/download_sample.py

# index your PDFs (anything in data/papers/)
python main.py ingest

# ask from the command line
python main.py ask "What problem does this paper solve?"

# ...or launch the web app (custom FastAPI backend + responsive chat frontend)
python server.py
#   then open http://127.0.0.1:8000
```

The web app has three modes — **Chat** (grounded, fact-checked answers), **Compare**
(cross-paper consensus/contradiction), and **Peer Review** (multi-agent reviewer
panel) — in a flat, editorial UI. Upload PDFs from the sidebar, then ask. Fully
responsive (the sidebar collapses on mobile).

---

## 🛡️ Rate limiting (protects your free credits)

The app caps LLM usage so it can't drain your free OpenRouter quota:

- **Min 3.5s between calls** (stays under OpenRouter's ~20 requests/min free limit)
- **40 calls/day** by default ≈ 20 questions (each question = answer + verify)
- **Automatic backoff** on 429 / network errors

Tune in `.env`: `MAX_CALLS_PER_DAY`, `MIN_SECONDS_BETWEEN_CALLS`, `MAX_RETRIES`.
The sidebar shows a live "LLM calls today" meter. (OpenRouter gives 50 free
requests/day, or 1000/day once you've purchased ≥ $10 of credit.)

---

## 📁 Project structure

```
.
├── server.py               # FastAPI backend + serves the frontend (primary UI)
├── frontend/               # custom responsive chat UI
│   ├── index.html
│   ├── styles.css          # dark glassmorphism, emerald/purple
│   └── app.js              # chat logic, markdown, uploads, export
├── main.py                 # command-line interface
├── config.py               # all settings in one place
├── requirements.txt
├── .env.example            # copy to .env and add your key
├── scripts/
│   └── download_sample.py  # fetch a sample arXiv paper
├── src/
│   ├── vectorstore.py      # ChromaDB + local embeddings
│   ├── ingest.py           # PDF -> chunks -> vector store
│   ├── retriever.py        # similarity search (+ optional re-ranking)
│   ├── llm.py              # Gemini client + answer generation
│   ├── verifier.py         # claim-by-claim grounding check
│   └── rag.py              # ties the pipeline together
└── data/
    ├── papers/             # your PDFs
    └── chroma/             # vector DB (auto-created)
```

---

## 🧠 Concepts to mention in interviews

- **RAG pipeline** — chunking strategy, embeddings, vector similarity search.
- **Grounding / hallucination detection** — the verification pass and why naive RAG
  is risky without it.
- **Structured outputs** — forcing the LLM to return a typed schema (Pydantic) for
  reliable, parseable verification results.
- **Cost engineering** — local embeddings + free LLM tier = $0 to run.

---

## 🔧 Ideas to extend it (great resume bullets)

- Add **cross-encoder re-ranking** (uncomment `sentence-transformers` in
  `requirements.txt`) and measure the retrieval-quality improvement.
- Add an **evaluation script** (e.g. answer faithfulness on a small test set).
- Support **arXiv search/import by ID** instead of manual upload.
- Highlight the **exact sentence** in each passage that supports a claim.
```

---

## Deploy

This app deploys as one FastAPI web service because `server.py` serves both the
API and `frontend/`.

### Render

1. Push this folder to a GitHub repository.
2. In Render, create a new Blueprint from the repo. The included `render.yaml`
   sets the build command, start command, health check, Python version, and a
   persistent disk for `data/`.
3. When Render prompts for environment variables, set:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-real-key
```

The deployed app starts with an empty index unless you upload PDFs in the UI or
copy existing data into the persistent disk. Do not commit `.env`, PDFs, or
`data/chroma/`; they are intentionally ignored.

### Manual Python Web Service

Use these settings on any Python host that provides a `PORT` variable:

```bash
Build command: pip install -r requirements.txt
Start command: uvicorn server:app --host 0.0.0.0 --port $PORT
```

Set `DATA_DIR` to a persistent storage path if you want uploaded PDFs and the
vector database to survive restarts and redeploys.
