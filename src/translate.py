"""Translate an indexed paper into another language.

Translates a representative slice of the paper's extracted text in one LLM call
(keeps it cheap on the free tier). Output is the translation only.
"""
from src.llm import _chat
from src.vectorstore import paper_context


def translate_paper(source: str, language: str) -> dict:
    text = paper_context(source, limit=20, max_chars=12000)
    if not text.strip():
        return {"error": f"No indexed content found for {source}."}

    system = (
        f"You are a professional translator. Translate the user's text into {language}. "
        "Preserve meaning, technical and scientific terms, and the structure "
        "(headings, lists, paragraphs). Output ONLY the translation — no preamble, "
        "no notes, no original text."
    )
    resp = _chat(
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": text}],
        temperature=0.2,
        max_tokens=4000,
    )
    return {
        "paper": source,
        "language": language,
        "text": (resp.choices[0].message.content or "").strip(),
    }
