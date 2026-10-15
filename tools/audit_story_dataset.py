#!/usr/bin/env python3
"""Audit the story dataset for genre balance and summary-like signals.

This script inspects the project's training text files and prints:
- per-genre counts
- whether the file is mostly plot-summary style
- rough summary-like heuristics
- sample low-quality rows that look like movie synopses instead of narrative prose
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "unused datasets" / "DataSet"
GENRES = ["<superhero>", "<action>", "<drama>", "<thriller>", "<horror>", "<sci_fi>"]


def parse_genre(line: str) -> str | None:
    for genre in GENRES:
        if genre in line:
            return genre
    return None


def is_summary_like(line: str) -> bool:
    text = line.strip()
    if not text:
        return True

    # Summary-like patterns often mention movie/story metadata and general recap language.
    summary_markers = [
        "movie begins",
        "the film",
        "the story follows",
        "in the end",
        "as the plot unfolds",
        "plot",
        "summary",
        "a thrilling story",
        "the movie",
        "the film follows",
        "the journey of",
        "is a",
        "when the story begins",
        "a dramatic story",
        "the characters",
        "at the beginning",
        "later the story",
        "the film is about",
    ]
    lowered = text.lower()
    marker_hits = sum(1 for marker in summary_markers if marker in lowered)

    # A lot of story-like lines are multi-sentence prose with dialogue or scene detail.
    sentence_count = max(1, len(re.split(r"(?<=[.!?])\s+", text)))
    if sentence_count >= 3 and marker_hits == 0:
        return False

    return marker_hits >= 2 or ("plot" in lowered and len(text.split()) > 25)


def sample_rows(path: Path, limit: int = 10):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in handle:
            cleaned = row.strip()
            if cleaned:
                rows.append(cleaned)
    return rows[:limit]


def audit_file(path: Path):
    counts = Counter()
    summary_like = 0
    total = 0
    suspicious_examples = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            cleaned = line.strip()
            if not cleaned:
                continue
            total += 1
            genre = parse_genre(cleaned)
            if genre:
                counts[genre] += 1
            if is_summary_like(cleaned):
                summary_like += 1
                if len(suspicious_examples) < 8:
                    suspicious_examples.append((line_number, genre or "unknown", cleaned[:220]))

    summary_ratio = summary_like / total if total else 0.0
    print(f"\n=== AUDIT: {path.name} ===")
    print(f"Total rows: {total}")
    print(f"Summary-like heuristic hits: {summary_like} ({summary_ratio:.2%})")
    print("Per-genre counts:")
    for genre in GENRES:
        print(f"  {genre}: {counts.get(genre, 0)}")
    print("\nSuspicious examples (first 8):")
    if suspicious_examples:
        for line_number, genre, sample in suspicious_examples:
            print(f"  [line {line_number}] [{genre}] {sample}")
    else:
        print("  none detected")


if __name__ == "__main__":
    targets = [
        DATASET_DIR / "6_genre_clean_training_data.txt",
        DATASET_DIR / "6_genre_balanced_training_data.txt",
        DATASET_DIR / "6_genre_eval_data.txt",
        DATASET_DIR / "college_student_stories.txt",
    ]

    for target in targets:
        if not target.exists():
            print(f"Missing file: {target}")
            continue
        audit_file(target)
