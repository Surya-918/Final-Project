from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_CANDIDATES = (
    PROJECT_ROOT / "trained model-college",
    PROJECT_ROOT / "trained model-retrained",
    PROJECT_ROOT / "trained model",
)
WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")
MISSING_WEIGHTS_MESSAGE = (
    "The college-trained tokenizer is in `trained model-college`, but the weight "
    "file is missing (need model.safetensors or pytorch_model.bin). The API will "
    "not fall back to base GPT-2, because that is why prompts looked untrained. "
    "Copy the weights from Google Drive folder "
    "`MyDrive/storytelling-college-retrained-2026-10-11` (or from a checkpoint-* "
    "folder / storytelling-college-model.zip) into `trained model-college`, then "
    "restart the backend. Helper: python copy_college_weights.py PATH_TO_FOLDER_OR_ZIP"
)


def has_weights(path: Path) -> bool:
    return any((path / name).is_file() for name in WEIGHT_NAMES)


def resolve_model_dir(path: Path) -> Optional[Path]:
    if has_weights(path):
        return path
    if not path.is_dir():
        return None
    checkpoints = sorted(
        (item for item in path.glob("checkpoint-*") if item.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for checkpoint in checkpoints:
        if has_weights(checkpoint):
            return checkpoint
    return None


def resolve_model_source() -> str:
    import os

    env_path = os.getenv("MODEL_NAME_OR_PATH", "").strip()
    if env_path:
        env_model = Path(env_path)
        if not env_model.is_absolute():
            env_model = (PROJECT_ROOT / env_model).resolve()
        resolved = resolve_model_dir(env_model)
        if resolved is not None:
            return str(resolved)
        if env_model.exists() and not env_model.is_dir():
            return env_path
        if (env_model / "config.json").is_file():
            raise FileNotFoundError(MISSING_WEIGHTS_MESSAGE)
        raise FileNotFoundError(
            f"MODEL_NAME_OR_PATH={env_path} has no model weights "
            f"(expected model.safetensors or pytorch_model.bin). "
            "Copy the Colab ZIP / Drive folder files into that directory."
        )
    missing_tokenizer_dirs = []
    for candidate in LOCAL_MODEL_CANDIDATES:
        resolved = resolve_model_dir(candidate)
        if resolved is not None:
            return str(resolved)
        if (candidate / "config.json").is_file():
            missing_tokenizer_dirs.append(str(candidate))
    if missing_tokenizer_dirs:
        raise FileNotFoundError(MISSING_WEIGHTS_MESSAGE)
    return "gpt2"
