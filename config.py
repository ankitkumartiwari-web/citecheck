"""Central configuration for the Research Paper Assistant.

Everything tunable lives here so you only edit one file. Values can be
overridden with environment variables (see .env.example).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file. override=True makes .env the source of
# truth even if a stale variable of the same name exists in the OS environment.
load_dotenv(override=True)

# ---- Paths ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data")).expanduser()
PAPERS_DIR = DATA_DIR / "papers"   # put your PDFs here
CHROMA_DIR = DATA_DIR / "chroma"   # vector DB is persisted here

PAPERS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ---- Accounts / sessions --------------------------------------------------
AUTH_DB = DATA_DIR / "auth.db"           # local user + session store (SQLite)
SESSION_COOKIE = os.getenv("SESSION_COOKIE", "cc_session")

# ---- Model (OpenRouter - free, OpenAI-compatible API) ---------------------
# Get free keys at https://openrouter.ai/keys and put them in .env.
# Supports MULTIPLE keys for load balancing - calls are round-robined across
# them and fail over on a rate limit, so two free keys ~= double the quota.
#   OPENROUTER_API_KEY=...           (primary)
#   OPENROUTER_API_KEY_2=...         (secondary)
#   OPENROUTER_API_KEYS=k1,k2,k3     (or a comma-separated list)
def _collect_keys():
    keys = []
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY_3"):
        v = (os.getenv(name) or "").strip()
        if v and v not in keys:
            keys.append(v)
    for v in (os.getenv("OPENROUTER_API_KEYS") or "").split(","):
        v = v.strip()
        if v and v not in keys:
            keys.append(v)
    if not keys:  # fall back to other common names only if no OpenRouter key set
        fb = (os.getenv("OPENAI_API_KEY") or os.getenv("OP") or "").strip()
        if fb:
            keys.append(fb)
    return keys


OPENROUTER_API_KEYS = _collect_keys()
OPENROUTER_API_KEY = OPENROUTER_API_KEYS[0] if OPENROUTER_API_KEYS else ""  # back-compat
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Free models on OpenRouter (note the ":free" suffix). Good alternatives:
#   openai/gpt-oss-120b:free
#   meta-llama/llama-3.3-70b-instruct:free
#   deepseek/deepseek-chat-v3-0324:free
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "openai/gpt-oss-120b:free")

# ---- Rate limiting (protect your free credits) ----------------------------
# Tuned for OpenRouter's free tier: ~20 requests/min and 50 requests/day PER KEY
# (1000/day per key if you've ever purchased >= $10 of credit). Limits scale with
# the number of keys, since calls are round-robined across them.
_N_KEYS = max(1, len(OPENROUTER_API_KEYS))
# Global throttle; per-key interval = this * N_KEYS, so we keep each key < 20/min.
MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("MIN_SECONDS_BETWEEN_CALLS", str(round(3.5 / _N_KEYS, 1))))
# Daily cap scales with keys (~40 calls/key/day, safely under 50).
MAX_CALLS_PER_DAY = int(os.getenv("MAX_CALLS_PER_DAY", str(40 * _N_KEYS)))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))  # backoff retries on 429/network

# ---- Vector store / retrieval --------------------------------------------
COLLECTION_NAME = "research_papers"
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 150      # overlap so sentences aren't cut mid-thought
TOP_K = 6                # how many passages to retrieve per question
