"""Command-line interface — handy for quick testing without the web UI.

Examples (run from the project root):
    python main.py ingest                         # ingest data/papers/*.pdf
    python main.py ask "What is the main result?"
"""
import sys

from src.ingest import ingest_dir
from src.rag import ask
from src.ratelimit import DailyLimitReached
from src.vectorstore import stats


def cmd_ingest():
    total, results = ingest_dir()
    for name, n in results.items():
        print(f"  {name}: {n} chunks")
    print(f"Done. {total} chunks total.")


def cmd_ask(question: str):
    if stats()["chunks"] == 0:
        print("No documents indexed. Run: python main.py ingest")
        return
    try:
        result = ask(question)
    except DailyLimitReached as e:
        print(f"\n[STOPPED] {e}")
        return
    print("\n=== ANSWER ===\n")
    print(result["answer"])

    report = result["report"]
    if report:
        flag = "GROUNDED" if report.overall_grounded else "SOME CLAIMS UNSUPPORTED"
        print(f"\n=== FACT-CHECK ({flag}) ===\n")
        print(report.summary, "\n")
        for c in report.claims:
            cites = ", ".join(f"[{i}]" for i in c.supporting_ids) or "-"
            print(f"[{c.verdict.value.upper()}] {c.claim}  (sources: {cites})")

    print("\n=== SOURCES ===")
    for c in result["chunks"]:
        loc = f"{c['source']}, p.{c['page']}" if c.get("page") else c["source"]
        print(f"  [{c['id']}] {loc}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    elif sys.argv[1] == "ingest":
        cmd_ingest()
    elif sys.argv[1] == "ask" and len(sys.argv) > 2:
        cmd_ask(" ".join(sys.argv[2:]))
    else:
        print(__doc__)
