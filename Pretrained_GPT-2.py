import logging
import math
import os
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("USE_TF", "0")

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import IntervalStrategy, get_last_checkpoint

from plot_graphs import save_training_metrics


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent


def format_story_example(text):
    match = re.match(r"^<BOS>\s+<([a-z0-9_]+)>\s+(.*?)\s+<EOS>\s*$", text)
    if not match:
        return text

    genre = match.group(1)
    story = match.group(2).strip()
    if story.lower().startswith("idea:"):
        story = re.sub(r"\\n(?=Story:)", "\n", story, count=1, flags=re.IGNORECASE)
        return f"<BOS> <{genre}> {story} <EOS>"

    boundary = re.search(r"[.!?](?:[\"')\]]*)\s+", story)
    if boundary:
        idea = story[: boundary.end()].strip()
        continuation = story[boundary.end() :].strip()
        if continuation:
            return f"<BOS> <{genre}> Idea: {idea}\nStory: {continuation} <EOS>"

    words = story.split()
    split_at = min(18, max(8, len(words) // 3)) if words else 0
    if split_at and len(words) > split_at:
        idea = " ".join(words[:split_at])
        continuation = " ".join(words[split_at:])
        return f"<BOS> <{genre}> Idea: {idea}\nStory: {continuation} <EOS>"

    return f"<BOS> <{genre}> Idea: {story}\nStory: {story} <EOS>"


def sync_special_token_ids(model, tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.story_prompt_format = "idea_story_v1"
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.bos_token_id = tokenizer.bos_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id
        generation_config.pad_token_id = tokenizer.pad_token_id
        if getattr(generation_config, "max_length", None) in (None, 20, 50):
            generation_config.max_length = getattr(model.config, "n_positions", 1024)


class LineTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_path, block_size, max_samples=0):
        self.examples = []
        story_marker = "\nStory:"
        file_paths = file_path if isinstance(file_path, (list, tuple)) else (file_path,)
        for current_path in file_paths:
            logger.info("Reading training examples from %s", current_path)
            with open(current_path, encoding="utf-8") as data_file:
                for row_number, line in enumerate(data_file, start=1):
                    if max_samples and len(self.examples) >= max_samples:
                        return
                    formatted_text = format_story_example(line.strip())
                    story_marker_index = formatted_text.find(story_marker)
                    if story_marker_index < 0:
                        logger.warning("Skipping row %s without a Story marker", row_number)
                        continue
                    token_ids = tokenizer.encode(
                        formatted_text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=block_size,
                    )
                    prompt_text = formatted_text[: story_marker_index + len(story_marker)]
                    prompt_token_count = len(
                        tokenizer.encode(prompt_text, add_special_tokens=False)
                    )
                    if len(token_ids) <= prompt_token_count:
                        logger.warning("Skipping row %s without story target tokens", row_number)
                        continue
                    labels = [-100] * prompt_token_count + token_ids[prompt_token_count:]
                    self.examples.append(
                        {
                            "input_ids": torch.tensor(token_ids, dtype=torch.long),
                            "attention_mask": torch.ones(len(token_ids), dtype=torch.long),
                            "labels": torch.tensor(labels, dtype=torch.long),
                        }
                    )
                    if row_number % 1000 == 0:
                        logger.info("Processed %s rows from %s", row_number, current_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class StoryDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = [feature["labels"] for feature in features]
        model_inputs = [
            {
                "input_ids": feature["input_ids"].tolist(),
                "attention_mask": feature["attention_mask"].tolist(),
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_inputs,
            padding=True,
            return_tensors="pt",
        )
        padded_labels = torch.full_like(batch["input_ids"], -100)
        for row_index, row_labels in enumerate(labels):
            row_length = row_labels.size(0)
            if self.tokenizer.padding_side == "left":
                padded_labels[row_index, -row_length:] = row_labels
            else:
                padded_labels[row_index, :row_length] = row_labels
        batch["labels"] = padded_labels
        return batch


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type such as gpt2."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    additional_train_data_file: Optional[str] = field(
        default=None, metadata={"help": "An optional additional input training data file."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )

    mlm: bool = field(
        default=False,
        metadata={
            "help": "Train with masked-language modeling loss instead of language modeling."
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


def get_dataset(args: DataTrainingArguments, tokenizer, evaluate=False, max_samples=0):
    file_path = (
        args.eval_data_file
        if evaluate
        else [path for path in (args.train_data_file, args.additional_train_data_file) if path]
    )
    return LineTextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=args.block_size,
        max_samples=max_samples,
    )


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_next_token_accuracy(eval_prediction):
    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids
    predicted_tokens = predictions[:, :-1]
    target_tokens = labels[:, 1:]
    valid_tokens = target_tokens != -100
    correct_tokens = (predicted_tokens == target_tokens) & valid_tokens
    valid_token_count = max(int(valid_tokens.sum()), 1)
    accuracy = correct_tokens.sum() / valid_token_count
    return {"next_token_accuracy": float(accuracy)}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    model_args = ModelArguments(
        model_name_or_path=os.getenv("TRAIN_MODEL_NAME_OR_PATH", "gpt2"),
        model_type="gpt2",
    )
    data_args = DataTrainingArguments(
        train_data_file=os.getenv(
            "TRAIN_DATA_FILE",
            str(PROJECT_ROOT / "hugging face dataset" / "training_ready" / "train.txt"),
        ),
        additional_train_data_file=os.getenv("ADDITIONAL_TRAIN_DATA_FILE", ""),
        eval_data_file=os.getenv(
            "EVAL_DATA_FILE",
            str(PROJECT_ROOT / "hugging face dataset" / "training_ready" / "validation.txt"),
        ),
        line_by_line=True,
        mlm=False,
        block_size=int(os.getenv("TRAIN_BLOCK_SIZE", "1024")),
        overwrite_cache=True,
    )
    evaluation_argument = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    training_args = TrainingArguments(
        output_dir=os.getenv(
            "TRAIN_OUTPUT_DIR", str(PROJECT_ROOT / "trained model-retrained")
        ),
        do_train=True,
        do_eval=True,
        do_predict=False,
        logging_steps=50,
        logging_first_step=True,
        per_device_train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "4")),
        per_device_eval_batch_size=int(os.getenv("TRAIN_EVAL_BATCH_SIZE", "4")),
        gradient_accumulation_steps=int(
            os.getenv("TRAIN_GRADIENT_ACCUMULATION_STEPS", "1")
        ),
        num_train_epochs=float(os.getenv("TRAIN_EPOCHS", "3")),
        save_total_limit=1,
        save_steps=int(os.getenv("TRAIN_SAVE_STEPS", "250")),
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    setattr(training_args, evaluation_argument, IntervalStrategy.EPOCH)
    output_dir = training_args.output_dir
    if output_dir is None:
        raise ValueError("Training output directory is required.")

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    if os.path.exists(output_dir) and os.listdir(output_dir) and not last_checkpoint and training_args.do_train:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) is not empty and has no resumable checkpoint. Choose another directory."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    logger.info("Loading model and tokenizer from %s", model_args.model_name_or_path)
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, cache_dir=model_args.cache_dir
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
    else:
        if model_args.model_type is None:
            raise ValueError("model_type is required when training from scratch.")
        config = AutoConfig.for_model(model_args.model_type)
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    special_tokens_dict = {
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",
        "additional_special_tokens": [
            "<superhero>",
            "<action>",
            "<drama>",
            "<thriller>",
            "<horror>",
            "<sci_fi>",
        ],
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    sync_special_token_ids(model, tokenizer)

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Get datasets

    max_train_samples = int(os.getenv("TRAIN_MAX_SAMPLES", "0"))
    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, max_samples=max_train_samples)
        if training_args.do_train
        else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
        if training_args.do_eval
        else None
    )
    if train_dataset is not None and not len(train_dataset):
        raise ValueError(f"No training examples were loaded from {data_args.train_data_file}.")
    if eval_dataset is not None and not len(eval_dataset):
        raise ValueError(f"No evaluation examples were loaded from {data_args.eval_data_file}.")
    logger.info(
        "Loaded %s training examples and %s evaluation examples.",
        len(train_dataset) if train_dataset is not None else 0,
        len(eval_dataset) if eval_dataset is not None else 0,
    )
    data_collator = StoryDataCollator(tokenizer=tokenizer)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_next_token_accuracy,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=last_checkpoint)
        sync_special_token_ids(trainer.model, tokenizer)
        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
            generation_config = getattr(trainer.model, "generation_config", None)
            if generation_config is not None:
                generation_config.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {
            "perplexity": perplexity,
            "next_token_accuracy": eval_output["eval_next_token_accuracy"],
        }

        output_eval_file = os.path.join(output_dir, "eval_results_lm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    if trainer.is_world_process_zero():
        metrics_csv, metrics_graph = save_training_metrics(
            trainer.state.log_history, Path(output_dir)
        )
        logger.info("Training metrics saved to %s", metrics_csv)
        logger.info("Training graph saved to %s", metrics_graph)

    return results


if __name__ == "__main__":
    main()
