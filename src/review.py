"""Multi-agent AI peer-review panel.

Three specialized reviewer agents independently critique a paper, then a "chair"
agent synthesizes their reviews into a recommendation. Each agent is one
rate-limited LLM call (3 reviewers + 1 chair = 4 calls per review).
"""
from src.llm import chat_json
from src.vectorstore import paper_context

REVIEWERS = [
    {
        "key": "methodology",
        "name": "Reviewer A — Methodology",
        "lens": "Soundness of methods & experiments",
        "system": "You are a rigorous methodology reviewer for a top conference. "
                  "Assess experimental design, baselines, ablations, statistical "
                  "rigor, and whether claims are supported by the experiments.",
    },
    {
        "key": "novelty",
        "name": "Reviewer B — Novelty & Significance",
        "lens": "Originality & impact",
        "system": "You are a reviewer focused on novelty and significance. Assess "
                  "how original the contribution is versus prior work and how much "
                  "it matters to the field.",
    },
    {
        "key": "clarity",
        "name": "Reviewer C — Clarity & Reproducibility",
        "lens": "Writing & reproducibility",
        "system": "You are a reviewer focused on clarity and reproducibility. Assess "
                  "how clearly the paper is written and whether someone could "
                  "reproduce the work from the description.",
    },
]

_REVIEW_SHAPE = (
    'Respond with ONLY this JSON (no prose): '
    '{"score": <integer 1-10>, "summary": "<2-3 sentences>", '
    '"strengths": ["<point>", ...], "weaknesses": ["<point>", ...]}'
)

_CHAIR_SYSTEM = (
    "You are the area chair. Given three reviews of a paper, synthesize a fair "
    "meta-review. Weigh the reviewers, resolve disagreements, and recommend a "
    "decision. Respond with ONLY this JSON: "
    '{"recommendation": "Accept|Weak Accept|Borderline|Weak Reject|Reject", '
    '"overall_score": <number 1-10>, "summary": "<3-4 sentences>", '
    '"top_strengths": ["..."], "top_concerns": ["..."]}'
)


def review_paper(source: str) -> dict:
    """Run the reviewer panel + chair synthesis for one paper."""
    context = paper_context(source)
    if not context.strip():
        return {"error": f"No indexed content found for {source}."}

    reviews = []
    for r in REVIEWERS:
        user = (f"Paper excerpts:\n\n{context}\n\n"
                f"Write your review. {_REVIEW_SHAPE}")
        data = chat_json(r["system"], user, max_tokens=900)
        reviews.append({
            "key": r["key"],
            "name": r["name"],
            "lens": r["lens"],
            "score": data.get("score"),
            "summary": data.get("summary", ""),
            "strengths": data.get("strengths", []) or [],
            "weaknesses": data.get("weaknesses", []) or [],
        })

    # Chair synthesis
    panel = "\n\n".join(
        f"{rv['name']} (score {rv['score']}/10): {rv['summary']}\n"
        f"Strengths: {'; '.join(rv['strengths'])}\n"
        f"Weaknesses: {'; '.join(rv['weaknesses'])}"
        for rv in reviews
    )
    chair = chat_json(_CHAIR_SYSTEM, f"Paper: {source}\n\nReviews:\n\n{panel}",
                      max_tokens=900)

    return {"paper": source, "reviewers": reviews, "chair": chair}
