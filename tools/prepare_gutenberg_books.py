#!/usr/bin/env python3
"""Prepare Gutenberg plain-text books as conservative action/drama examples.

Each book stays wholly in either train or validation to reduce leakage. Genre
assignments are provisional: action is represented by adventure fiction, while
the drama books are literary novels rather than verified genre annotations.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = ROOT / "unused datasets" / "newly dataset" / "books"
OUTPUT_DIR = BOOKS_DIR / "prepared"
SOURCE_FILE = BOOKS_DIR / "book_sources.json"
START_RE = re.compile(r"^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK\b", re.I)
END_RE = re.compile(r"^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK\b", re.I)
CHAPTER_RE = re.compile(
    r"^(?:(?:PART|BOOK|CHAPTER|STAVE|ACT)\s+(?:[IVXLCDM]+|\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|"
    r"SEVEN|EIGHT|NINE|TEN)(?:\b|[\s.:—-])|"
    r"(?:THE\s+)?(?:OLD BUCCANEER|SEA-COOK|MY SHORE ADVENTURE|THE STOCKADE|"
    r"MY SEA ADVENTURE|CAPTAIN SILVER)\s*$)",
    re.I,
)
CHAPTER_HEADING_RE = re.compile(
    r"^\s*(?:(?:CHAPTER|PART|BOOK|STAVE|ACT)\s+"
    r"(?:[IVXLCDM]+|\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)"
    r"(?:[.\]\s:-]+.*)?)\s*$",
    re.I,
)
ROMAN_HEADING_RE = re.compile(r"^\s*[IVXLCDM]{1,5}(?:[.)])?\s*$")
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
SENTENCE_END_RE = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+")
MAX_WORDS = 230
MIN_FINAL_WORDS = 35

def load_books(source_file: Path = SOURCE_FILE) -> tuple[dict, ...]:
    """Load curated source metadata while keeping every whole book in one split."""
    sources = json.loads(source_file.read_text(encoding="utf-8"))
    books = []
    seen_ids: set[int] = set()
    for source in sources["books"]:
        ebook_id = int(source["id"])
        genre = source["genre"]
        if ebook_id in seen_ids:
            raise ValueError(f"Duplicate Project Gutenberg ebook ID: {ebook_id}")
        if genre not in {"action", "drama"}:
            raise ValueError(f"Unsupported genre {genre!r} for ebook {ebook_id}.")
        if source["split"] not in {"train", "validation"}:
            raise ValueError(f"Unsupported split for ebook {ebook_id}.")
        seen_ids.add(ebook_id)
        relative_path = Path(
            source.get("file", f"{genre}/pg-{ebook_id}.txt")
        )
        books.append(
            {
                "id": ebook_id,
                "path": relative_path,
                "genre": genre,
                "split": source["split"],
                "title": source["title"],
                "url": f"https://www.gutenberg.org/ebooks/{ebook_id}",
                "start_marker": source.get("start_marker"),
                "end_marker": source.get("end_marker"),
                "genre_note": (
                    "adventure and action-adjacent fiction used as a provisional "
                    "action proxy"
                    if genre == "action"
                    else "literary, social, or relationship-centered fiction used "
                    "as a provisional drama proxy"
                ),
            }
        )
    if not books:
        raise ValueError(f"No book records found in {source_file}")
    return tuple(books)


BOOKS = load_books()


def extract_book_text(raw_text: str) -> str:
    """Remove Gutenberg wrapper text using its explicit boundary markers."""
    lines = raw_text.lstrip("\ufeff").splitlines()
    start = next((i for i, line in enumerate(lines) if START_RE.match(line.strip())), None)
    end = next((i for i, line in enumerate(lines) if END_RE.match(line.strip())), None)
    if start is None or end is None or end <= start:
        raise ValueError("Missing or invalid Project Gutenberg start/end markers.")

    body_lines = lines[start + 1 : end]
    return unicodedata.normalize("NFC", "\n".join(body_lines)).replace("\r", "")


def remove_front_matter(body: str, start_marker: str | None = None) -> str:
    """Start at the first real chapter prose, skipping contents and title pages."""
    if start_marker:
        position = body.casefold().find(start_marker.casefold())
        if position < 0:
            raise ValueError(f"Configured narrative start marker not found: {start_marker!r}")
        line_start = body.rfind("\n", 0, position) + 1
        return body[line_start:]

    lines = body.splitlines()
    contents_index = next(
        (
            i
            for i, line in enumerate(lines)
            if re.fullmatch(r"\s*(?:table of )?contents[.:]?\s*", line, re.I)
        ),
        -1,
    )
    search_start = contents_index + 1 if contents_index >= 0 else 0
    numbered_candidates = [
        index
        for index in range(search_start, len(lines))
        if _is_numbered_section_heading(lines[index])
        and not _looks_like_contents_row(lines[index])
    ]
    for opening_only in (True, False):
        for index in numbered_candidates:
            if opening_only and not _is_opening_section_heading(lines[index]):
                continue
            prose_start = _prose_after_heading(lines, index)
            if prose_start is not None:
                return "\n".join(lines[prose_start:])

    if contents_index >= 0:
        for index in range(contents_index + 1, len(lines)):
            line = lines[index].strip()
            if not _is_named_section_heading(line) or _is_numbered_section_heading(line):
                continue
            if _looks_like_contents_row(line):
                continue
            prose_start = _prose_after_heading(lines, index, break_on_named_headings=True)
            if prose_start is not None:
                return "\n".join(lines[prose_start:])

    raise ValueError("Could not identify the first chapter's narrative prose.")


def remove_source_tail(body: str, end_marker: str | None = None) -> str:
    """Remove embedded Project Gutenberg footer and edition-promotion text."""
    if end_marker:
        position = body.casefold().find(end_marker.casefold())
        if position < 0:
            raise ValueError(f"Configured narrative end marker not found: {end_marker!r}")
        body = body[:position].rstrip()

    tail_markers = (
        r"End of Project Gutenberg(?:['’]s)?",
        r"Did you enjoy this Editor's Cut(?:\(tm\))? edition\?",
        r"These are left in for the next proofer to turn into CAPS for PG\.",
        r"This only occurs twice in the book, so both are left in\.",
        r"\[24\]\s+It is suggested to me that this book is",
        r"The Outlook:",
        r"A Note on the Text",
        r"The Age of Innocence first appeared in four large installments",
        r"(?im)^\s*=*\s*Transcriber['’]s Notes?\s*[:=]*\s*$",
        r"(?im)^\s*AUTHORITIES\s*$",
    )
    positions = [
        match.start()
        for marker in tail_markers
        if (match := re.search(marker, body, re.I))
    ]
    return body[: min(positions)].rstrip() if positions else body


def _is_numbered_section_heading(text: str) -> bool:
    text = text.strip()
    if CHAPTER_HEADING_RE.match(text) or ROMAN_HEADING_RE.match(text):
        return True
    if re.match(r"^\d{1,3}[.)]\s+\S", text):
        return True
    return bool(
        text == text.upper()
        and re.match(r"^[IVXLCDM]{1,5}[.)]\s+\S", text)
    )


def _is_opening_section_heading(text: str) -> bool:
    text = text.strip()
    named_number = re.match(
        r"^(?:CHAPTER|PART|BOOK|STAVE|ACT)\s+(?:I\b|1\b|ONE\b)",
        text,
        re.I,
    )
    if named_number:
        return True
    if ROMAN_HEADING_RE.fullmatch(text.upper()) and text.upper().rstrip(".)") == "I":
        return True
    return bool(re.match(r"^(?:1[.)]\s+|I[.)]\s+\S)", text, re.I))


def _looks_like_contents_row(text: str) -> bool:
    numbered_entries = re.findall(
        r"\b(?:CHAPTER|PART|BOOK|STAVE|ACT)\s+(?:[IVXLCDM]+|\d+)\b"
        r"|(?:^|\s)[IVXLCDM]{1,5}[.)]\s+\S"
        r"|(?:^|\s)\d{1,3}[.)]\s+\S",
        text,
        re.I,
    )
    return (
        len(WORD_RE.findall(text)) > 15
        or len(numbered_entries) > 1
        or bool(re.search(r"\bPAGE\b", text, re.I))
        or len(re.findall(r"[—–]", text)) > 1
        or re.search(r"\.{2,}|\s{3,}", text) is not None
    )


def _prose_after_heading(
    lines: list[str],
    heading_index: int,
    *,
    break_on_named_headings: bool = False,
) -> int | None:
    paragraph_start = None
    paragraph_lines: list[str] = []
    first_paragraph_start = None
    for candidate_index in range(heading_index + 1, min(len(lines), heading_index + 30)):
        candidate = lines[candidate_index].strip()
        if not candidate:
            if paragraph_lines:
                paragraph = " ".join(paragraph_lines)
                if len(WORD_RE.findall(paragraph)) >= 18 and not paragraph.isupper():
                    return first_paragraph_start or paragraph_start
                if first_paragraph_start is None:
                    first_paragraph_start = paragraph_start
                paragraph_lines = []
                paragraph_start = None
            continue
        if (
            _is_numbered_section_heading(candidate)
            or (
                break_on_named_headings
                and _is_named_section_heading(candidate)
            )
            or re.search(r"\.{2,}", candidate)
        ):
            return None
        if paragraph_start is None:
            paragraph_start = candidate_index
        paragraph_lines.append(candidate)

    if paragraph_lines:
        paragraph = " ".join(paragraph_lines)
        if len(WORD_RE.findall(paragraph)) >= 18 and not paragraph.isupper():
            return first_paragraph_start or paragraph_start
    return None


def _is_section_heading(text: str) -> bool:
    """Recognize numbered and all-caps section headings without dialogue false positives."""
    text = text.strip()
    if not text or text.casefold() in {
        "contents",
        "table of contents",
        "preface",
        "introduction",
        "transcriber's note",
        "transcribers note",
        "summary",
    }:
        return False
    if _is_numbered_section_heading(text):
        return True
    words = WORD_RE.findall(text)
    return len(words) <= 12 and text.isupper()


def _is_named_section_heading(text: str) -> bool:
    text = text.strip()
    if _is_section_heading(text):
        return True
    words = WORD_RE.findall(text)
    return bool(
        1 <= len(words) <= 8
        and all(word[:1].isupper() for word in words)
    )


def _is_heading_or_page_number(paragraph: str) -> bool:
    text = paragraph.strip()
    if not text or text.startswith("[Illustration"):
        return True
    if re.fullmatch(r"[\s*#=_-]{3,}", text):
        return True
    words = WORD_RE.findall(text)
    if re.fullmatch(r"\d{1,4}", text):
        return True
    if CHAPTER_RE.match(text) or _is_section_heading(text):
        return True
    if len(words) <= 25 and re.search(r"--", text) and not re.search(r"[.!?]\s", text):
        return True
    if len(words) <= 25 and text.isupper():
        return True
    if re.search(r"\.{4,}", text):
        return True
    return False


def _split_long_paragraph(paragraph: str) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_END_RE.split(paragraph) if part.strip()]
    if not sentences:
        return []
    segments: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        words = len(WORD_RE.findall(sentence))
        if words > MAX_WORDS:
            if current:
                segments.append(" ".join(current))
                current, current_words = [], 0
            token_segment: list[str] = []
            token_segment_words = 0
            segment_count = (words + MAX_WORDS - 1) // MAX_WORDS
            target_words = (words + segment_count - 1) // segment_count
            sentence_segments = 0
            for token in sentence.split():
                token_words = len(WORD_RE.findall(token))
                if token_words > MAX_WORDS:
                    raise ValueError("A whitespace-delimited token exceeds the chunk limit.")
                should_balance = (
                    sentence_segments < segment_count - 1
                    and token_segment_words + token_words > target_words
                )
                if token_segment and (
                    token_segment_words + token_words > MAX_WORDS or should_balance
                ):
                    segments.append(" ".join(token_segment))
                    sentence_segments += 1
                    token_segment, token_segment_words = [], 0
                token_segment.append(token)
                token_segment_words += token_words
            if token_segment:
                segments.append(" ".join(token_segment))
            continue
        if current and current_words + words > MAX_WORDS:
            segments.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += words
    if current:
        segments.append(" ".join(current))
    return segments


def chunk_book_text(body: str) -> list[str]:
    """Make single-line, paragraph-aware chunks that fit the trainer's block size."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if _is_heading_or_page_number(stripped):
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))

    chunks: list[str] = []
    pending: list[str] = []
    pending_words = 0
    for paragraph in paragraphs:
        paragraph_words = len(WORD_RE.findall(paragraph))
        if paragraph_words > MAX_WORDS:
            if pending:
                chunks.append(" ".join(pending))
                pending, pending_words = [], 0
            chunks.extend(_split_long_paragraph(paragraph))
            continue
        if pending and pending_words + paragraph_words > MAX_WORDS:
            chunks.append(" ".join(pending))
            pending, pending_words = [], 0
        pending.append(paragraph)
        pending_words += paragraph_words
    if pending:
        chunks.append(" ".join(pending))

    coalesced: list[str] = []
    index = 0
    while index < len(chunks):
        chunk = chunks[index]
        if len(WORD_RE.findall(chunk)) < MIN_FINAL_WORDS:
            if index + 1 < len(chunks):
                combined = f"{chunk} {chunks[index + 1]}"
                if len(WORD_RE.findall(combined)) <= MAX_WORDS:
                    coalesced.append(combined)
                    index += 2
                    continue
            if coalesced:
                combined = f"{coalesced[-1]} {chunk}"
                if len(WORD_RE.findall(combined)) <= MAX_WORDS:
                    coalesced[-1] = combined
                    index += 1
                    continue
        coalesced.append(chunk)
        index += 1
    return [
        re.sub(r"\s+", " ", chunk).strip()
        for chunk in coalesced
        if chunk.strip()
    ]


def prepare(
    output_dir: Path,
    books_dir: Path = BOOKS_DIR,
    books: tuple[dict, ...] = BOOKS,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_lines: dict[str, list[str]] = {"train": [], "validation": []}
    manifest_books = []

    for book in books:
        path = books_dir / book["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing source book: {path}")
        raw = path.read_text(encoding="utf-8-sig")
        body = remove_front_matter(
            extract_book_text(raw), start_marker=book.get("start_marker")
        )
        body = remove_source_tail(body, end_marker=book.get("end_marker"))
        chunks = chunk_book_text(body)
        if not chunks:
            raise ValueError(f"No narrative text extracted from {path}")
        word_counts = [len(WORD_RE.findall(chunk)) for chunk in chunks]
        if max(word_counts) > MAX_WORDS:
            raise ValueError(f"Chunk exceeds {MAX_WORDS} words: {path}")
        split_lines[book["split"]].extend(
            f"<BOS> <{book['genre']}> {chunk} <EOS>\n" for chunk in chunks
        )
        manifest_books.append(
            {
                "gutenberg_id": book["id"],
                "title": book["title"],
                "source_file": str(book["path"]),
                "source_url": book["url"],
                "genre": book["genre"],
                "genre_note": book["genre_note"],
                "story_start_marker": book.get("start_marker"),
                "story_end_marker": book.get("end_marker"),
                "split": book["split"],
                "chunks": len(chunks),
                "words": sum(word_counts),
                "min_chunk_words": min(word_counts),
                "max_chunk_words": max(word_counts),
            }
        )

    for split, lines in split_lines.items():
        if not lines:
            raise ValueError(f"No chunks generated for {split}.")
        (output_dir / f"gutenberg_{split}.txt").write_text(
            "".join(lines), encoding="utf-8", newline="\n"
        )

    counts = Counter()
    for book in manifest_books:
        counts[(book["split"], book["genre"])] += book["chunks"]
    manifest = {
        "format": "<BOS> <genre> narrative chunk <EOS>",
        "max_chunk_words": MAX_WORDS,
        "source_license_note": (
            "Check the individual Project Gutenberg ebook page/license and local law. "
            "Action tags are based on adventure/action-adjacent sources, and Drama "
            "tags on literary/social fiction; neither is a verified annotation."
        ),
        "books": manifest_books,
        "counts": {
            f"{split}_{genre}": count
            for (split, genre), count in sorted(counts.items())
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books-dir", type=Path, default=BOOKS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--source-file", type=Path, default=SOURCE_FILE)
    args = parser.parse_args()
    manifest = prepare(args.output_dir, args.books_dir, load_books(args.source_file))

    for split in ("train", "validation"):
        path = args.output_dir / f"gutenberg_{split}.txt"
        print(f"{split}: {path.stat().st_size:,} bytes")
        for genre in ("action", "drama"):
            count = manifest["counts"].get(f"{split}_{genre}", 0)
            print(f"  {genre}: {count:,} chunks")
    print(f"Manifest: {args.output_dir / 'manifest.json'}")
    print(
        "Validation split is by whole book. Genre assignments remain provisional; "
        "review outputs before training."
    )


if __name__ == "__main__":
    main()
