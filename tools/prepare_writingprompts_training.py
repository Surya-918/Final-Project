#!/usr/bin/env python3
"""Clean WritingPrompts genre candidates into trainer-ready text files."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
import re
import unicodedata
from contextlib import ExitStack
from pathlib import Path
from typing import Any

try:
    from tokenizers import Tokenizer
except ImportError as exc:
    raise SystemExit(
        "The tokenizers package is required to validate the trainer's token limit. "
        "Install it with `python -m pip install tokenizers`."
    ) from exc

try:
    from better_profanity import profanity
except ImportError as exc:
    raise SystemExit(
        "better-profanity is required to remove profanity from training rows. "
        "Install it with `python -m pip install better-profanity`."
    ) from exc

profanity.load_censor_words()
PROFANITY_WORDS = frozenset(str(word).casefold() for word in profanity.CENSOR_WORDSET)
MAX_PROFANITY_PHRASE_WORDS = max(len(word.split()) for word in PROFANITY_WORDS)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "unused datasets" / "hugging face dataset" / "segregated_premise_review"
DEFAULT_OUTPUT_DIR = ROOT / "unused datasets" / "hugging face dataset" / "training_ready_premise_review"
DEFAULT_TOKENIZER = ROOT / "trained model-college"
GENRES = ("action", "drama")
SPLITS = ("train", "validation", "test")
MIN_PROMPT_WORDS = 5
MIN_STORY_WORDS = 100
MAX_TOKENS = 1024
ACCEPTED_SIGNAL_STRENGTHS = {"medium", "high"}
BATCH_SIZE = 256

WP_PREFIX = re.compile(r"^\s*\[\s*wp\s*\]\s*", re.IGNORECASE)
NON_STORY_PROMPT = re.compile(
    r"^\s*(?:\[\s*ot\s*\]\s*|writing\s+(?:workshop|challenge)\b|workshop\s*#)",
    re.IGNORECASE,
)
MINOR_CONTEXT = re.compile(
    r"\b(?:minor|underage|child(?:ren)?|kid(?:s)?|son|daughter|boy|girl|teen(?:ager)?|"
    r"high\s+school|middle\s+school|schoolgirl|schoolboy|\d{1,2}[- ]year[- ]old)\b",
    re.IGNORECASE,
)
HIGH_RISK_MINOR_CONTEXT = re.compile(
    r"\b(?:minor|underage|child(?:ren)?|kid(?:s)?|teen(?:ager)?|adolescent|"
    r"high\s+school|middle\s+school|schoolgirl|schoolboy|\d{1,2}[- ]year[- ]old)\b",
    re.IGNORECASE,
)
SEXUAL_CONTENT = re.compile(
    r"\b(?:sex(?:ual|ually)?|whore|testicl\w*|genital\w*|penis|vagina|"
    r"masturb\w*|molest\w*|handjob|orgasm\w*|porn(?:ography)?|fondl\w*|"
    r"rape\w*|incest|pedophil\w*|mating|copulat\w*|aphrodisiac\w*|"
    r"sexy|seduc\w*|erotic\w*|horny|arous\w*|lust\w*|nude|naked|"
    r"make\s+out|lower\s+regions|stroke\s+\w+\s+off)\b",
    re.IGNORECASE,
)
SEXUAL_VIOLENCE = re.compile(
    r"\b(?:rape\w*|sexual\s+assault|molest\w*|pedophil\w*|incest\w*|"
    r"sex\s+traffick\w*|non[- ]consensual)\b",
    re.IGNORECASE,
)
DIRECT_ABUSE = re.compile(
    r"\b(?:idiots?|morons?|stupid|dumb(?:ass)?|losers?|jerks?|"
    r"scumbags?|imbeciles?|cretins?|fools?)\b",
    re.IGNORECASE,
)
WHITESPACE = re.compile(r"\s+")
WORD = re.compile(r"\b[\w'-]+\b", re.UNICODE)
POST_NOISE = re.compile(
    r"\b(?:"
    r"constructive feedback|"
    r"this is my first (?:wp|writing ?prompts?) post|"
    r"thanks for reading|"
    r"thank you for reading|"
    r"i do not own (?:these|the|any)|"
    r"i don't own (?:these|the|any)|"
    r"no copyright infringement intended|"
    r"not my character|"
    r"would love feedback|"
    r"forgive my english|"
    r"sorry if (?:my english|it sucks)|"
    r"i apologize for the utter destruction above|"
    r"homer must be spinning in his grave|"
    r"the prompt contained two of my pet peeves|"
    r"edit\s*:\s*(?:wrote|this|formatting|spelling)|"
    r"first prompt response|"
    r"not a complete story|"
    r"this is my first time and english is not my main language|"
    r"i wrote this as if|"
    r"my ending (?:got|is|was) .{0,40}?(?:lazy|unfinished)|"
    r"check out /r/|"
    r"more stuff at my subreddit|"
    r"i have no idea wtf i(?:'|’)m actually doing"
    r")\b",
    re.IGNORECASE,
)


def clean_text(text: str, *, is_prompt: bool = False) -> str | None:
    """Normalize whitespace and reject text with replacement/control noise."""
    if "\ufffd" in text:
        return None
    if any(
        unicodedata.category(character) == "Cc" and character not in "\n\r\t"
        for character in text
    ):
        return None
    cleaned = html.unescape(text)
    if is_prompt:
        cleaned = WP_PREFIX.sub("", cleaned, count=1)
    return WHITESPACE.sub(" ", cleaned).strip()


def normalized_prompt_key(prompt: str) -> str:
    """Build a stable key for exact normalized-prompt deduplication."""
    prompt = WP_PREFIX.sub("", prompt, count=1)
    normalized = " ".join(WORD.findall(prompt.casefold()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def normalized_story_key(story: str) -> str:
    """Build a stable key for exact normalized-story deduplication."""
    normalized = " ".join(WORD.findall(story.casefold()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_non_story_prompt(prompt: str) -> bool:
    return bool(NON_STORY_PROMPT.search(prompt))


def contains_post_noise(text: str) -> bool:
    return bool(POST_NOISE.search(text))


def contains_abusive_language(text: str) -> bool:
    tokens = [word.casefold() for word in WORD.findall(text)]
    for start in range(len(tokens)):
        for width in range(1, min(MAX_PROFANITY_PHRASE_WORDS, len(tokens) - start) + 1):
            if " ".join(tokens[start : start + width]) in PROFANITY_WORDS:
                return True
    return bool(SEXUAL_VIOLENCE.search(text) or DIRECT_ABUSE.search(text))


def contains_minor_sexual_content(text: str, related_text: str = "") -> bool:
    sexual_positions = [match.start() for match in SEXUAL_CONTENT.finditer(text)]
    if any(
        abs(sexual_position - minor_match.start()) <= 180
        for minor_match in MINOR_CONTEXT.finditer(text)
        for sexual_position in sexual_positions
    ):
        return True
    return bool(
        related_text
        and HIGH_RISK_MINOR_CONTEXT.search(text)
        and SEXUAL_CONTENT.search(related_text[:1200])
    )


def load_tokenizer(tokenizer_file: Path):
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    tokenizer.no_truncation()
    tokenizer.add_special_tokens(
        [
            "<BOS>",
            "<EOS>",
            "<PAD>",
            "<superhero>",
            "<action>",
            "<drama>",
            "<thriller>",
            "<horror>",
            "<sci_fi>",
        ]
    )
    return tokenizer


def format_example(genre: str, prompt: str, story: str) -> str:
    if genre not in GENRES:
        raise ValueError(f"Unsupported training genre: {genre}")
    return f"<BOS> <{genre}> Idea: {prompt}\nStory: {story} <EOS>"


def serialize_example(genre: str, prompt: str, story: str) -> str:
    """Keep each trainer example on one physical line."""
    return format_example(genre, prompt, story).replace("\n", r"\n")


def _source_files(input_dir: Path) -> list[tuple[str, str, Path]]:
    sources = []
    for split in SPLITS:
        for genre in GENRES:
            path = input_dir / f"{genre}_{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"Required candidate file not found: {path}")
            sources.append((split, genre, path))
    return sources


def _read_candidates(path: Path, split: str, genre: str, counters: Counter):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            counters[(split, genre, "input")] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name}:{line_number}") from exc

            if (
                record.get("genre_candidate") != genre
                or record.get("source_split") != split
                or record.get("keyword_signal_strength")
                not in ACCEPTED_SIGNAL_STRENGTHS
            ):
                counters[(split, genre, "low_signal_or_mismatch")] += 1
                continue

            prompt = record.get("prompt")
            story = record.get("story")
            if not isinstance(prompt, str) or not isinstance(story, str):
                counters[(split, genre, "invalid_fields")] += 1
                continue
            if is_non_story_prompt(prompt):
                counters[(split, genre, "non_story_prompt")] += 1
                continue

            prompt = clean_text(prompt, is_prompt=True)
            story = clean_text(story)
            if not prompt or not story:
                counters[(split, genre, "text_noise")] += 1
                continue
            if contains_abusive_language(prompt) or contains_abusive_language(story):
                counters[(split, genre, "abusive_or_sexual_violence")] += 1
                continue
            if contains_minor_sexual_content(prompt, story) or contains_minor_sexual_content(story):
                counters[(split, genre, "minor_sexual_content")] += 1
                continue
            if contains_post_noise(prompt) or contains_post_noise(story):
                counters[(split, genre, "post_or_meta_noise")] += 1
                continue
            if len(WORD.findall(prompt)) < MIN_PROMPT_WORDS:
                counters[(split, genre, "short_prompt")] += 1
                continue
            if len(WORD.findall(story)) < MIN_STORY_WORDS:
                counters[(split, genre, "short_story")] += 1
                continue

            yield prompt, story


def _process_batch(
    batch: list[tuple[str, str]],
    genre: str,
    split: str,
    tokenizer,
    output,
    combined_output,
    seen_prompts: set[str],
    seen_stories: set[str],
    counters: Counter,
) -> None:
    candidates = []
    batch_prompts: set[str] = set()
    batch_stories: set[str] = set()
    for prompt, story in batch:
        key = normalized_prompt_key(prompt)
        if key in seen_prompts or key in batch_prompts:
            counters[(split, genre, "duplicate_prompt")] += 1
            continue
        story_key = normalized_story_key(story)
        if story_key in seen_stories or story_key in batch_stories:
            counters[(split, genre, "duplicate_story")] += 1
            continue
        batch_prompts.add(key)
        batch_stories.add(story_key)
        candidates.append((key, story_key, prompt, story))

    if not candidates:
        return

    formatted = [
        format_example(genre, prompt, story)
        for _, _, prompt, story in candidates
    ]
    encoded = tokenizer.encode_batch(formatted, add_special_tokens=False)
    for (key, story_key, prompt, story), text, encoding in zip(
        candidates, formatted, encoded
    ):
        if len(encoding.ids) > MAX_TOKENS:
            counters[(split, genre, "over_token_limit")] += 1
            continue
        seen_prompts.add(key)
        seen_stories.add(story_key)
        serialized = serialize_example(genre, prompt, story) + "\n"
        output.write(serialized)
        combined_output.write(serialized)
        counters[(split, genre, "written")] += 1


def prepare(
    input_dir: Path,
    output_dir: Path,
    tokenizer_path: Path,
) -> dict[str, Any]:
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer path does not exist: {tokenizer_path}. "
            "Pass --tokenizer with the same GPT-2 tokenizer used for training."
        )

    tokenizer_file = (
        tokenizer_path / "tokenizer.json"
        if tokenizer_path.is_dir()
        else tokenizer_path
    )
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"Tokenizer JSON not found: {tokenizer_file}")
    tokenizer = load_tokenizer(tokenizer_file)

    sources = _source_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counters: Counter = Counter()
    seen_prompts: set[str] = set()
    seen_stories: set[str] = set()

    with ExitStack() as stack:
        outputs = {
            (split, genre): stack.enter_context(
                (output_dir / f"{genre}_{split}.txt").open(
                    "w", encoding="utf-8", newline="\n"
                )
            )
            for split, genre, _ in sources
        }
        combined_outputs = {
            split: stack.enter_context(
                (output_dir / f"{split}.txt").open(
                    "w", encoding="utf-8", newline="\n"
                )
            )
            for split in SPLITS
        }
        for split, genre, path in sources:
            batch: list[tuple[str, str]] = []
            for candidate in _read_candidates(path, split, genre, counters):
                batch.append(candidate)
                if len(batch) >= BATCH_SIZE:
                    _process_batch(
                        batch,
                        genre,
                        split,
                        tokenizer,
                        outputs[(split, genre)],
                        combined_outputs[split],
                        seen_prompts,
                        seen_stories,
                        counters,
                    )
                    batch.clear()
            if batch:
                _process_batch(
                    batch,
                    genre,
                    split,
                    tokenizer,
                    outputs[(split, genre)],
                    combined_outputs[split],
                    seen_prompts,
                    seen_stories,
                    counters,
                )

    report = {
        "method": (
            "Medium/high premise-keyword candidates; reject rows with profanity "
            "from better-profanity, sexual-violence terms, or selected direct insults; "
            "also reject explicit/suggestive sexual content near minor references; "
            "normalized paired text; "
            f"minimum {MIN_PROMPT_WORDS} prompt words and {MIN_STORY_WORDS} story "
            f"words; exact prompt and normalized-story deduplication across splits; "
            "near-duplicate stories are not automatically detected; no truncation; "
            f"maximum {MAX_TOKENS} tokens using the selected local GPT-2 tokenizer. "
            "Genre candidates are premise-keyword heuristics and not human-verified."
        ),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "tokenizer": str(tokenizer_path),
        "limits": {
            "max_tokens": MAX_TOKENS,
            "min_prompt_words": MIN_PROMPT_WORDS,
            "min_story_words": MIN_STORY_WORDS,
            "keyword_signal_strength_included": sorted(
                ACCEPTED_SIGNAL_STRENGTHS
            ),
        },
        "counts": {
            split: {
                genre: {
                    reason: counters[(split, genre, reason)]
                    for reason in (
                        "input",
                        "written",
                        "low_signal_or_mismatch",
                        "invalid_fields",
                        "non_story_prompt",
                        "abusive_or_sexual_violence",
                        "minor_sexual_content",
                        "text_noise",
                        "post_or_meta_noise",
                        "short_prompt",
                        "short_story",
                        "duplicate_prompt",
                        "duplicate_story",
                        "over_token_limit",
                    )
                    if counters[(split, genre, reason)]
                }
                for genre in GENRES
            }
            for split in SPLITS
        },
        "written_total": {
            split: sum(counters[(split, genre, "written")] for genre in GENRES)
            for split in SPLITS
        },
    }
    (output_dir / "preparation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()
    report = prepare(args.input_dir, args.output_dir, args.tokenizer)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
