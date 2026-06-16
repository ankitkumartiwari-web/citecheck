"""LLM layer (OpenRouter — free, OpenAI-compatible API).

Responsibilities:
  - format retrieved passages into a context block
  - run rate-limited, retrying chat calls (`_chat`)
  - answer a question using ONLY that context, with citations

OpenRouter speaks the OpenAI API, so we use the official `openai` SDK and just
point its base_url at OpenRouter. Every call goes through `_chat`, which enforces
the rate limits in src/ratelimit.py and backs off on 429s.
"""
import json
import re
import threading
import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

import config
from src import ratelimit

# Optional headers OpenRouter uses for app attribution (not required).
_HEADERS = {"X-Title": "CiteCheck"}

# One client per API key. Calls round-robin across them to balance load, and
# fail over to the next key on a rate limit.
_clients = None
_rr_lock = threading.Lock()
_rr_index = 0


def _get_clients():
    global _clients
    if _clients is None:
        if not config.OPENROUTER_API_KEYS:
            raise RuntimeError(
                "No API key found. Add OPENROUTER_API_KEY to .env "
                "(get a free key at https://openrouter.ai/keys)."
            )
        _clients = [
            OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=k)
            for k in config.OPENROUTER_API_KEYS
        ]
    return _clients


def _next_client():
    """Return the next client in round-robin order (thread-safe)."""
    global _rr_index
    clients = _get_clients()
    with _rr_lock:
        client = clients[_rr_index % len(clients)]
        _rr_index += 1
    return client


def get_client():
    """Back-compat: return the first client."""
    return _get_clients()[0]


def _chat(messages, temperature=0.2, max_tokens=1500, response_format=None):
    """One rate-limited chat completion with backoff.

    - Counts against the daily cap exactly once (via ratelimit.guard()).
    - Retries 429 / network errors with exponential backoff.
    - If the model rejects `response_format` (JSON mode), retries once without it.
    """
    ratelimit.guard()  # daily cap + min-interval throttle (may raise / sleep)

    kwargs = dict(
        model=config.ANSWER_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers=_HEADERS,
    )
    if response_format is not None:
        kwargs["response_format"] = response_format

    delay = 2.0
    for attempt in range(config.MAX_RETRIES):
        # Round-robin per attempt: balances load and fails over across keys.
        client = _next_client()
        try:
            return client.chat.completions.create(**kwargs)
        except BadRequestError:
            # Most likely the model doesn't support JSON mode — drop it and retry.
            if "response_format" in kwargs:
                kwargs.pop("response_format")
                continue
            raise
        except (RateLimitError, APIConnectionError, APITimeoutError):
            if attempt == config.MAX_RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("LLM call failed after retries.")


def extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response (handles ```json fences)."""
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def chat_json(system: str, user: str, max_tokens: int = 1500) -> dict:
    """Run one rate-limited JSON-mode chat and return the parsed object."""
    resp = _chat(
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return extract_json(resp.choices[0].message.content or "")


def format_context(chunks) -> str:
    """Render passages as numbered blocks the model can cite by id."""
    blocks = []
    for c in chunks:
        loc = f"{c['source']}, p.{c['page']}" if c.get("page") else c["source"]
        blocks.append(f"[{c['id']}] ({loc})\n{c['text']}")
    return "\n\n".join(blocks)


ANSWER_SYSTEM = (
    "You are a research-paper assistant. Answer the user's question using ONLY the "
    "provided context passages from academic papers. Cite every passage you rely on "
    "with its numeric id in square brackets, e.g. [1] or [2][3]. "
    "If the context does not contain the answer, say so plainly instead of guessing. "
    "Be precise, concise, and neutral."
)


def answer(query: str, chunks) -> str:
    """Generate a grounded, cited answer for `query` from `chunks`."""
    context = format_context(chunks)
    resp = _chat(
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user",
             "content": f"Context passages:\n\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
        max_tokens=1500,
    )
    return (resp.choices[0].message.content or "").strip()
