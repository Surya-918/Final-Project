"""Copy a Colab/Drive checkpoint or ZIP into `trained model-college`."""

import json
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEST = PROJECT_ROOT / "trained model-college"
WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")
COPY_NAMES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "eval_results_lm.txt",
    "training_metrics.csv",
    "training_metrics.png",
]


def has_weights(path: Path) -> bool:
    return any((path / name).is_file() for name in WEIGHT_NAMES)


def find_source(path: Path) -> Path:
    if path.is_file() and path.suffix.lower() == ".zip":
        extract_to = PROJECT_ROOT / ".college-weights-unpack"
        if extract_to.exists():
            shutil.rmtree(extract_to)
        extract_to.mkdir()
        with zipfile.ZipFile(path) as archive:
            archive.extractall(extract_to)
        path = extract_to
    if has_weights(path):
        return path
    checkpoints = sorted(
        (item for item in path.glob("checkpoint-*") if item.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for checkpoint in checkpoints:
        if has_weights(checkpoint):
            return checkpoint
    nested = [item for item in path.rglob("config.json") if has_weights(item.parent)]
    if nested:
        return nested[0].parent
    raise FileNotFoundError(
        f"No model.safetensors or pytorch_model.bin under {path}"
    )


def patch_eos(config_path: Path) -> None:
    if not config_path.is_file():
        return
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["bos_token_id"] = 50257
    data["eos_token_id"] = 50258
    data["pad_token_id"] = 50259
    data["story_prompt_format"] = "idea_story_v1"
    config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python copy_college_weights.py "
            "C:\\path\\to\\storytelling-college-retrained-2026-10-11"
        )
        print("   or: python copy_college_weights.py C:\\path\\to\\storytelling-college-model.zip")
        sys.exit(1)
    source = find_source(Path(sys.argv[1]).expanduser().resolve())
    DEST.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in COPY_NAMES:
        file_path = source / name
        if file_path.is_file():
            shutil.copy2(file_path, DEST / name)
            copied.append(name)
    patch_eos(DEST / "config.json")
    gen_cfg = DEST / "generation_config.json"
    if gen_cfg.is_file():
        data = json.loads(gen_cfg.read_text(encoding="utf-8"))
        data["bos_token_id"] = 50257
        data["eos_token_id"] = 50258
        data["pad_token_id"] = 50259
        gen_cfg.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if not has_weights(DEST):
        raise FileNotFoundError("Copy finished but destination still has no weight file.")
    print(f"Copied {len(copied)} files into {DEST}")
    print("Restart the backend: npm run backend")


if __name__ == "__main__":
    main()
