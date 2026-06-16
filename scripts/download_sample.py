"""Download a sample paper so you can try the app immediately.

Grabs the classic "Attention Is All You Need" (the Transformer paper) from arXiv.

    python scripts/download_sample.py
"""
import sys
import urllib.request
from pathlib import Path

# Make the project root importable when run as `python scripts/download_sample.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

SAMPLE_URL = "https://arxiv.org/pdf/1706.03762"
DEST = config.PAPERS_DIR / "attention_is_all_you_need.pdf"


def main():
    if DEST.exists():
        print(f"Already downloaded: {DEST}")
        return
    print(f"Downloading sample paper -> {DEST} ...")
    req = urllib.request.Request(SAMPLE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(DEST, "wb") as f:
        f.write(r.read())
    print("Done. Now run:  python main.py ingest")


if __name__ == "__main__":
    main()
