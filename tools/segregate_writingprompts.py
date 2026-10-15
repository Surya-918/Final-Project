#!/usr/bin/env python3
"""Split WritingPrompts Parquet files into Action, Drama, and review sets.

Genre assignment is a transparent keyword heuristic, not a human-verified label.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import duckdb
except ImportError as exc:
    raise SystemExit(
        "DuckDB is required to read Parquet files. Install it with "
        "`python -m pip install duckdb`."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "unused datasets" / "hugging face dataset" / "source parquet"
DEFAULT_OUTPUT_DIR = ROOT / "unused datasets" / "hugging face dataset" / "segregated_premise_review"

ACTION_CUES = (
    r"\baction(?:-packed)?\b",
    r"\battack(?:ed|s|ing)?\b",
    r"\barmy\b",
    r"\bassassin(?:s|ated|ation)?\b",
    r"\bbattle(?:field|s|d|ing)?\b",
    r"\bbomb(?:s|ed|ing)?\b",
    r"\bbrawl(?:s|ed|ing)?\b",
    r"\bbullet(?:s)?\b",
    r"\bchase(?:s|d|ing)?\b",
    r"\bcombat\b",
    r"\bcommando(?:s)?\b",
    r"\bcriminal(?:s)?\b",
    r"\bdetonat(?:e|es|ed|ing|ion)\b",
    r"\bduel(?:s|ed|ing)?\b",
    r"\bexplod(?:e|es|ed|ing|sion)\b",
    r"\bfight(?:s|ing|er|ers)?\b",
    r"\bgun(?:s|fire)?\b",
    r"\bheist(?:s)?\b",
    r"\bhostage(?:s)?\b",
    r"\binvad(?:e|es|ed|ing|er|ers|ion)\b",
    r"\bkill(?:s|ed|ing)?\b",
    r"\bmission(?:s)?\b",
    r"\bpolice\b",
    r"\bpursu(?:e|es|ed|ing|it)\b",
    r"\braid(?:s|ed|ing)?\b",
    r"\brescu(?:e|es|ed|ing)\b",
    r"\brob(?:bery|ber|bers|bed|bing)\b",
    r"\bsoldier(?:s)?\b",
    r"\bshoot(?:s|ing|er|ers)?\b",
    r"\bspy(?:ies)?\b",
    r"\bsurviv(?:e|es|ed|ing|or|ors|al)\b",
    r"\bsword(?:s)?\b",
    r"\bterrorist(?:s)?\b",
    r"\bweapon(?:s)?\b",
    r"\bwar(?:s|fare|rior|riors)?\b",
)

DRAMA_CUES = (
    r"\baffair(?:s)?\b",
    r"\bargument(?:s)?\b",
    r"\bbetray(?:al|als|ed|ing|s)?\b",
    r"\bboyfriend(?:s)?\b",
    r"\bbreakup(?:s)?\b",
    r"\bchildhood\b",
    r"\bcustody\b",
    r"\bdaughter(?:s)?\b",
    r"\bdivorc(?:e|es|ed|ing)\b",
    r"\bemotional(?:ly)?\b",
    r"\bestrang(?:e|ed|ement)\b",
    r"\bfamil(?:y|ies)\b",
    r"\bfather(?:s)?\b",
    r"\bforgiv(?:e|es|eness|en|ing)\b",
    r"\bfriendship(?:s)?\b",
    r"\bgrief\b",
    r"\bheartbreak(?:s|ing)?\b",
    r"\bhusband(?:s)?\b",
    r"\bidentity\b",
    r"\billness(?:es)?\b",
    r"\blonel(?:y|iness)\b",
    r"\blove(?:s|d|ly|less)?\b",
    r"\bmarri(?:age|ages|ed|es)\b",
    r"\bmemories\b",
    r"\bmother(?:s)?\b",
    r"\bmourn(?:s|ed|ing)?\b",
    r"\bparent(?:s|hood)?\b",
    r"\bregret(?:s|ted|ting|ful)?\b",
    r"\breunion(?:s)?\b",
    r"\brelationship(?:s)?\b",
    r"\bsibling(?:s)?\b",
    r"\bterminal illness\b",
    r"\bwedding(?:s)?\b",
    r"\bwife\b",
    r"\bwives\b",
)

def _combined_pattern(prefix: str, patterns: tuple[str, ...]) -> str:
    return "|".join(
        f"(?P<{prefix}{index}>{pattern})" for index, pattern in enumerate(patterns)
    )


GENRE_CUE_PATTERN = re.compile(
    _combined_pattern("action", ACTION_CUES)
    + "|"
    + _combined_pattern("drama", DRAMA_CUES),
    re.IGNORECASE,
)
SPLITS = ("train", "validation", "test")


def genre_scores(prompt: str, story: str) -> tuple[int, int]:
    """Count distinct genre cues in the premise, not the generated story."""
    matched_cues = {
        match.lastgroup for match in GENRE_CUE_PATTERN.finditer(prompt)
    }
    action_score = sum(cue.startswith("action") for cue in matched_cues)
    drama_score = sum(cue.startswith("drama") for cue in matched_cues)
    return int(action_score), int(drama_score)


def classify_genre(prompt: str, story: str) -> tuple[str, str, int, int]:
    """Return a tentative genre, confidence band, and both heuristic scores."""
    action_score, drama_score = genre_scores(prompt, story)
    return classify_scores(action_score, drama_score)


def classify_scores(action_score: int, drama_score: int) -> tuple[str, str, int, int]:
    """Assign a tentative genre from the two distinct-cue counts."""
    winner, score, other = (
        ("action", action_score, drama_score)
        if action_score > drama_score
        else ("drama", drama_score, action_score)
    )
    margin = score - other
    if score < 2 or margin < 1:
        return "unclear", "low", action_score, drama_score
    confidence = "high" if score >= 5 and margin >= 3 else "medium"
    return winner, confidence, action_score, drama_score


def source_split(path: Path) -> str:
    match = re.match(r"(train|validation|test)-", path.name)
    if not match:
        raise ValueError(f"Unrecognized dataset split filename: {path.name}")
    return match.group(1)


def _write_record(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def segregate(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[tuple[str, str]] = Counter()
    total_rows = 0
    rejected_rows = 0
    action_pattern = "(?:" + "|".join(ACTION_CUES) + ")"
    drama_pattern = "(?:" + "|".join(DRAMA_CUES) + ")"

    with ExitStack() as stack:
        outputs = {
            (split, genre): stack.enter_context(
                (output_dir / f"{genre}_{split}.jsonl").open("w", encoding="utf-8")
            )
            for split in SPLITS
            for genre in ("action", "drama", "unclear")
        }
        with duckdb.connect() as connection:
            relation = connection.execute(
                """
                SELECT
                    prompt,
                    story,
                    filename,
                    len(list_distinct(regexp_extract_all(
                        lower(coalesce(prompt, '')),
                        ?, 0
                    ))) AS action_score,
                    len(list_distinct(regexp_extract_all(
                        lower(coalesce(prompt, '')),
                        ?, 0
                    ))) AS drama_score
                FROM read_parquet(?, filename = true)
                """,
                [action_pattern, drama_pattern, [str(path) for path in files]],
            )
            while rows := relation.fetchmany(500):
                for prompt, story, filename, action_score, drama_score in rows:
                    total_rows += 1
                    if not isinstance(prompt, str) or not prompt.strip():
                        rejected_rows += 1
                        continue
                    if not isinstance(story, str) or not story.strip():
                        rejected_rows += 1
                        continue

                    genre, confidence, action_score, drama_score = classify_scores(
                        action_score, drama_score
                    )
                    file_path = Path(filename)
                    split = source_split(file_path)
                    record = {
                        "prompt": prompt.strip(),
                        "story": story.strip(),
                        "genre_candidate": genre,
                        "keyword_signal_strength": confidence,
                        "source_split": split,
                        "source_file": file_path.name,
                        "action_score": action_score,
                        "drama_score": drama_score,
                    }
                    _write_record(outputs[(split, genre)], record)
                    counts[(split, genre)] += 1

    report = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "method": (
            "Rule-based keyword triage over each premise only. "
            "Genre candidates and signal strength are not human-verified genre "
            "labels or quality judgments."
        ),
        "total_rows": total_rows,
        "rejected_empty_rows": rejected_rows,
        "counts": {
            split: {
                genre: counts[(split, genre)]
                for genre in ("action", "drama", "unclear")
            }
            for split in SPLITS
        },
        "total_by_genre": {
            genre: sum(counts[(split, genre)] for split in SPLITS)
            for genre in ("action", "drama", "unclear")
        },
    }
    (output_dir / "segregation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = segregate(args.input_dir, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
