"""data/qa_pairs/loader.py — load Q&A pairs from CSV."""

import csv
from pathlib import Path

QA_CSV = Path(__file__).parent / "qa_pairs.csv"


def load_qa_pairs(path: Path = QA_CSV) -> list[dict]:
    """Load Q&A pairs from CSV. Returns list of {question, ground_truth} dicts."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            {"question": row["question"], "ground_truth": row["ground_truth"]}
            for row in reader
            if row.get("question")
        ]
