"""
Ingestion: loads .txt standard documents from data/, splits them into
clause-level chunks (so citations can point to a specific clause number),
and returns a flat list of chunk dicts ready for indexing.

Swap in real BIS PDFs later by dropping extracted text into data/ with the
same "Clause N: Title" structure, or adjust CLAUSE_PATTERN below.
"""

import os
import re

from nlp_pipeline import DOC_CATEGORY

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CLAUSE_PATTERN = re.compile(r"((?:Clause|Section)\s+\d+[:.]\s*.+)")


def _doc_title(text: str, fallback: str) -> str:
    first_line = text.strip().splitlines()[0].strip()
    return first_line if first_line else fallback


def load_chunks():
    """Returns a list of dicts: {doc, standard_id, clause, text}."""
    chunks = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(DATA_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        title = _doc_title(text, fname)
        standard_id = title.split(":")[0].strip()
        category = DOC_CATEGORY.get(fname, "standard")

        # Split on "Clause N: ..." headers, keeping the header with its body
        parts = CLAUSE_PATTERN.split(text)
        # parts[0] is the preamble/title before the first clause
        preamble = parts[0].strip()
        if preamble:
            chunks.append({
                "doc": fname,
                "standard_id": standard_id,
                "category": category,
                "clause": "Preamble",
                "text": preamble,
            })

        # remaining parts alternate: header, body, header, body...
        for i in range(1, len(parts) - 1, 2):
            header = parts[i].strip()
            body = parts[i + 1].strip()
            clause_label = header.split(":")[0].strip()
            chunks.append({
                "doc": fname,
                "standard_id": standard_id,
                "category": category,
                "clause": clause_label,
                "text": f"{header}\n{body}",
            })

    return chunks


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {DATA_DIR}")
    for c in chunks[:3]:
        print("-" * 40)
        print(c["standard_id"], "|", c["clause"])
        print(c["text"][:150])
