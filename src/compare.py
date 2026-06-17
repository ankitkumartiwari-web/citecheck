"""Cross-paper contradiction & consensus finder.

Given a claim, gather the most relevant passages from EACH indexed paper, then
ask the model (one call) to judge each paper's stance - supports / refutes /
neutral - with a short evidence quote, and summarize the overall consensus.
"""
from src.llm import chat_json
from src.vectorstore import query_in_paper, stats

_SYSTEM = (
    "You compare how different research papers relate to a specific claim. "
    "For each paper, decide its stance on the claim based ONLY on its passages:\n"
    "  - 'supports': the paper provides evidence FOR the claim\n"
    "  - 'refutes': the paper provides evidence AGAINST the claim\n"
    "  - 'neutral': the paper is related but does not clearly support or refute it\n"
    "Quote a short snippet as evidence. Respond with ONLY this JSON:\n"
    '{"stances": [{"paper": "<filename>", "stance": "supports|refutes|neutral", '
    '"evidence": "<short quote>", "explanation": "<one sentence>"}], '
    '"consensus_summary": "<2-3 sentences on overall agreement/disagreement>"}'
)


def compare_claim(claim: str, k: int = 3, papers=None, user: str | None = None) -> dict:
    """Judge each selected paper's stance on `claim` and summarize consensus.

    Only `user`'s papers are considered. If `papers` is given, the comparison is
    further narrowed to those; otherwise all of the user's indexed papers.
    """
    indexed = stats(user)["papers"]
    papers = [p for p in indexed if (papers is None or p in papers)]
    if len(papers) < 2:
        return {"error": "Select at least two papers to compare across them."}

    blocks = []
    considered = []
    for src in papers:
        passages = query_in_paper(src, claim, user, k=k)
        if not passages:
            continue
        considered.append(src)
        joined = "\n".join(f"- {p}" for p in passages)
        blocks.append(f"### {src}\n{joined}")

    if not blocks:
        return {"error": "No relevant passages found for that claim."}

    user = (f"Claim: {claim}\n\n"
            f"Passages grouped by paper:\n\n" + "\n\n".join(blocks))
    data = chat_json(_SYSTEM, user, max_tokens=1600)

    stances = data.get("stances", []) or []
    counts = {"supports": 0, "refutes": 0, "neutral": 0}
    for s in stances:
        st = s.get("stance")
        if st in counts:
            counts[st] += 1

    return {
        "claim": claim,
        "papers_considered": considered,
        "stances": stances,
        "counts": counts,
        "consensus_summary": data.get("consensus_summary", ""),
    }
