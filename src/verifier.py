"""Citation / grounding verification - the feature that sets this project apart.

After the model writes an answer, we send the answer + the same context back and
ask the model to act as a strict fact-checker: break the answer into atomic claims
and mark each as supported / partially_supported / unsupported against the
passages. This surfaces hallucinations instead of trusting the answer blindly.

Free OpenRouter models vary in how well they support strict structured output, so
we ask for a JSON object, then parse + validate it with Pydantic and fall back
gracefully if a model returns malformed JSON.
"""
import json
import re
from enum import Enum
from typing import List

from pydantic import BaseModel, ValidationError

from src.llm import _chat, format_context


class Verdict(str, Enum):
    supported = "supported"
    partially_supported = "partially_supported"
    unsupported = "unsupported"


class ClaimCheck(BaseModel):
    claim: str
    verdict: Verdict
    supporting_ids: List[int] = []   # ids of passages that back the claim
    rationale: str = ""              # one short sentence explaining the verdict


class VerificationReport(BaseModel):
    claims: List[ClaimCheck] = []
    overall_grounded: bool = False
    summary: str = ""


# Exact JSON shape we want back - embedded in the prompt so any model can comply.
_SCHEMA_HINT = (
    '{"claims":[{"claim":"<text>","verdict":"supported|partially_supported|'
    'unsupported","supporting_ids":[1,2],"rationale":"<one sentence>"}],'
    '"overall_grounded":true,"summary":"<one sentence>"}'
)

VERIFY_SYSTEM = (
    "You are a strict, skeptical fact-checker. You receive numbered context passages "
    "from research papers and an answer generated from them. Break the answer into "
    "atomic factual claims. For each claim decide:\n"
    "  - 'supported': fully backed by one or more passages\n"
    "  - 'partially_supported': partly backed but overstated or missing key detail\n"
    "  - 'unsupported': not backed by any passage (a likely hallucination)\n"
    "List the ids of the passages that support each claim. If a claim is not clearly "
    "present in the passages, mark it unsupported. Set overall_grounded to false if "
    "any claim is unsupported.\n\n"
    "Respond with ONLY a JSON object of exactly this shape (no markdown, no prose):\n"
    + _SCHEMA_HINT
)


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response (handles ```json fences)."""
    if not text:
        return {}
    # Strip code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        # Otherwise grab from the first { to the last }.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def verify(answer_text: str, chunks) -> VerificationReport:
    """Fact-check `answer_text` against `chunks`; return a structured report.

    Goes through llm._chat, so it shares the rate limiting, backoff, and the
    automatic fallback when a free model doesn't support JSON mode.
    """
    context = format_context(chunks)
    resp = _chat(
        messages=[
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user",
             "content": f"Context passages:\n\n{context}\n\n"
                        f"Answer to fact-check:\n{answer_text}"},
        ],
        temperature=0.0,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content or ""
    data = _extract_json(content)
    try:
        return VerificationReport.model_validate(data)
    except ValidationError:
        return VerificationReport(
            claims=[],
            overall_grounded=False,
            summary="The model returned an output that couldn't be parsed for "
                    "verification. Try asking again or switch to another free model.",
        )
