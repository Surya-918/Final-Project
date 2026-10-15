import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = PROJECT_ROOT / "Pretrained_GPT-2.py"
TRAINER_SPEC = importlib.util.spec_from_file_location("story_training_module", TRAINER_PATH)
if TRAINER_SPEC is None or TRAINER_SPEC.loader is None:
    raise ImportError(f"Could not load trainer module from {TRAINER_PATH}")
TRAINER_MODULE = importlib.util.module_from_spec(TRAINER_SPEC)
sys.modules[TRAINER_SPEC.name] = TRAINER_MODULE
TRAINER_SPEC.loader.exec_module(TRAINER_MODULE)


class CharacterTokenizer:
    padding_side = "left"

    def encode(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
    ):
        token_ids = list(range(1, len(text) + 1))
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return token_ids

    def pad(self, features, padding=True, return_tensors="pt"):
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_rows = []
        attention_rows = []
        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            padding_length = max_length - len(input_ids)
            input_rows.append([0] * padding_length + input_ids)
            attention_rows.append([0] * padding_length + attention_mask)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
        }


class StoryTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.train_file = Path(self.temp_dir.name) / "train.txt"
        self.train_file.write_text(
            "<BOS> <drama> Idea: A family reunites.\\nStory: They forgive. <EOS>\n"
            "<BOS> <action> Idea: A rescue begins.\\nStory: They escape safely. <EOS>\n",
            encoding="utf-8",
        )
        self.tokenizer = CharacterTokenizer()
        self.dataset = TRAINER_MODULE.LineTextDataset(
            tokenizer=self.tokenizer,
            file_path=str(self.train_file),
            block_size=512,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_masks_prompt_and_keeps_story_targets(self):
        example = self.dataset[0]
        formatted_text = TRAINER_MODULE.format_story_example(
            "<BOS> <drama> Idea: A family reunites.\\nStory: They forgive. <EOS>"
        )
        prompt_text = formatted_text.split("\nStory:", 1)[0] + "\nStory:"
        prompt_token_count = len(self.tokenizer.encode(prompt_text))

        self.assertTrue(torch.all(example["labels"][:prompt_token_count] == -100))
        self.assertTrue(
            torch.equal(
                example["labels"][prompt_token_count:],
                example["input_ids"][prompt_token_count:],
            )
        )

    def test_collator_keeps_left_padding_and_prompt_labels_masked(self):
        batch = TRAINER_MODULE.StoryDataCollator(self.tokenizer)(
            [self.dataset[0], self.dataset[1]]
        )

        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        for row_index, example in enumerate(self.dataset.examples):
            padding_length = batch["input_ids"].shape[1] - len(example["input_ids"])
            self.assertTrue(torch.all(batch["attention_mask"][row_index, :padding_length] == 0))
            self.assertTrue(torch.all(batch["labels"][row_index, :padding_length] == -100))
            active_labels = batch["labels"][row_index] != -100
            self.assertTrue(torch.any(active_labels))


if __name__ == "__main__":
    unittest.main()