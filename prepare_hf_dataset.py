"""Convert and clean Hugging Face r/WritingPrompts segregated dataset into the
exact structured prompt format required by Pretrained_GPT-2.py:

    <BOS> <genre> Idea: {premise} Story: {story} <EOS>

Outputs balanced Action and Drama training and evaluation files in DataSet/.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def clean_prompt(text: str) -> str:
    """Strip Reddit tags like [ WP ], [ TT ], etc. and clean punctuation."""
    # Remove leading tag e.g. [ WP ], [WP], [ TT ], [EU], [CW], [PI], [RF]
    cleaned = re.sub(r"^\s*\[\s*[A-Za-z0-9_ -]+\s*\]\s*", "", text)
    cleaned = re.sub(r"^\s*\(\s*[A-Za-z0-9_ -]+\s*\)\s*", "", cleaned)
    # Fix tokenized contractions
    cleaned = re.sub(r"\bn\s*'\s*t\b", "n't", cleaned)
    cleaned = re.sub(r"\b'\s*s\b", "'s", cleaned)
    cleaned = re.sub(r"\b'\s*re\b", "'re", cleaned)
    cleaned = re.sub(r"\b'\s*ve\b", "'ve", cleaned)
    cleaned = re.sub(r"\b'\s*m\b", "'m", cleaned)
    cleaned = re.sub(r"\b'\s*ll\b", "'ll", cleaned)
    cleaned = re.sub(r"\b'\s*d\b", "'d", cleaned)
    # Fix spaces before punctuation
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    # Quotes
    cleaned = re.sub(r"``\s*", '"', cleaned)
    cleaned = re.sub(r"\s*''", '"', cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_story(text: str, max_words: int = 420) -> str:
    """Normalize whitespace, detokenize, and truncate at sentence boundary."""
    # Fix tokenized contractions
    cleaned = re.sub(r"\bn\s*'\s*t\b", "n't", text)
    cleaned = re.sub(r"\b'\s*s\b", "'s", cleaned)
    cleaned = re.sub(r"\b'\s*re\b", "'re", cleaned)
    cleaned = re.sub(r"\b'\s*ve\b", "'ve", cleaned)
    cleaned = re.sub(r"\b'\s*m\b", "'m", cleaned)
    cleaned = re.sub(r"\b'\s*ll\b", "'ll", cleaned)
    cleaned = re.sub(r"\b'\s*d\b", "'d", cleaned)
    # Fix spaces before punctuation
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    # Quotes
    cleaned = re.sub(r"``\s*", '"', cleaned)
    cleaned = re.sub(r"\s*''", '"', cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Truncate to max_words at sentence boundary if possible
    words = cleaned.split()
    if len(words) > max_words:
        truncated = " ".join(words[:max_words])
        # Find last sentence boundary
        last_period = max(
            truncated.rfind(". "),
            truncated.rfind("! "),
            truncated.rfind("? "),
            truncated.rfind('."'),
            truncated.rfind('!"'),
            truncated.rfind('?"'),
        )
        if last_period > int(len(truncated) * 0.7):
            cleaned = truncated[: last_period + 1].strip()
        else:
            cleaned = truncated + "..."
    return cleaned


def load_segregated_file(file_path: Path, max_samples: int = 0):
    """Load rows prioritizing 'high' signal over 'medium' signal."""
    high_items = []
    medium_items = []
    other_items = []

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = clean_prompt(item.get("prompt", ""))
            story = clean_story(item.get("story", ""))

            # Filter out trivially short entries
            if len(prompt.split()) < 3 or len(story.split()) < 30:
                continue

            record = {
                "prompt": prompt,
                "story": story,
                "signal": item.get("keyword_signal_strength", "medium"),
            }

            if record["signal"] == "high":
                high_items.append(record)
            elif record["signal"] == "medium":
                medium_items.append(record)
            else:
                other_items.append(record)

    all_sorted = high_items + medium_items + other_items
    if max_samples > 0:
        return all_sorted[:max_samples]
    return all_sorted


def convert_dataset(
    source_dir: Path,
    output_dir: Path,
    train_samples_per_genre: int = 15000,
    eval_samples_per_genre: int = 1000,
    seed: int = 42,
):
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Converting Hugging Face WritingPrompts dataset...")
    print(f"Source: {source_dir}")
    print(f"Destination: {output_dir}")
    print(f"Train samples per genre: {train_samples_per_genre}")
    print(f"Eval samples per genre:  {eval_samples_per_genre}")
    print("=" * 60)

    # 1. Load Training Data
    action_train_path = source_dir / "action_train.jsonl"
    drama_train_path = source_dir / "drama_train.jsonl"

    print("\n[1/4] Loading Action train data...")
    action_train = load_segregated_file(action_train_path, train_samples_per_genre)
    print(f"  -> Loaded {len(action_train)} Action training examples.")

    print("[2/4] Loading Drama train data...")
    drama_train = load_segregated_file(drama_train_path, train_samples_per_genre)
    print(f"  -> Loaded {len(drama_train)} Drama training examples.")

    # Format training lines
    train_lines = []
    for item in action_train:
        train_lines.append(
            f"<BOS> <action> Idea: {item['prompt']} Story: {item['story']} <EOS>\n"
        )
    for item in drama_train:
        train_lines.append(
            f"<BOS> <drama> Idea: {item['prompt']} Story: {item['story']} <EOS>\n"
        )

    # Shuffle to interleave genres uniformly
    random.shuffle(train_lines)

    train_out_file = output_dir / "hf_action_drama_train.txt"
    with train_out_file.open("w", encoding="utf-8") as f:
        f.writelines(train_lines)
    print(
        f"\n[SUCCESS] Training file written: {train_out_file} ({len(train_lines)} lines, {train_out_file.stat().st_size / (1024**2):.2f} MB)"
    )

    # 2. Load Validation Data
    action_val_path = source_dir / "action_validation.jsonl"
    drama_val_path = source_dir / "drama_validation.jsonl"

    print("\n[3/4] Loading Action validation data...")
    action_val = load_segregated_file(action_val_path, eval_samples_per_genre)
    print(f"  -> Loaded {len(action_val)} Action validation examples.")

    print("[4/4] Loading Drama validation data...")
    drama_val = load_segregated_file(drama_val_path, eval_samples_per_genre)
    print(f"  -> Loaded {len(drama_val)} Drama validation examples.")

    eval_lines = []
    for item in action_val:
        eval_lines.append(
            f"<BOS> <action> Idea: {item['prompt']} Story: {item['story']} <EOS>\n"
        )
    for item in drama_val:
        eval_lines.append(
            f"<BOS> <drama> Idea: {item['prompt']} Story: {item['story']} <EOS>\n"
        )

    random.shuffle(eval_lines)

    eval_out_file = output_dir / "hf_action_drama_eval.txt"
    with eval_out_file.open("w", encoding="utf-8") as f:
        f.writelines(eval_lines)
    print(
        f"[SUCCESS] Evaluation file written: {eval_out_file} ({len(eval_lines)} lines, {eval_out_file.stat().st_size / (1024**2):.2f} MB)"
    )

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE!")
    print(f"Train dataset: {train_out_file.name} ({len(train_lines):,} examples)")
    print(f"Eval dataset:  {eval_out_file.name} ({len(eval_lines):,} examples)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face segregated dataset into structured prompt training format."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "unused datasets"
        / "hugging face dataset"
        / "segregated",
        help="Path to segregated JSONL directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "unused datasets" / "DataSet",
        help="Output directory for converted TXT files.",
    )
    parser.add_argument(
        "--train-samples-per-genre",
        type=int,
        default=15000,
        help="Number of training examples per genre (default: 15000 -> 30,000 total).",
    )
    parser.add_argument(
        "--eval-samples-per-genre",
        type=int,
        default=1000,
        help="Number of validation examples per genre (default: 1000 -> 2,000 total).",
    )
    args = parser.parse_args()

    convert_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_samples_per_genre=args.train_samples_per_genre,
        eval_samples_per_genre=args.eval_samples_per_genre,
    )
