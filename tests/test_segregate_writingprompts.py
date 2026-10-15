import unittest
from pathlib import Path

from tools.segregate_writingprompts import classify_genre, source_split


class WritingPromptsSegregationTests(unittest.TestCase):
    def test_classifies_action_with_multiple_distinct_signals(self):
        genre, confidence, action_score, drama_score = classify_genre(
            "A soldier must rescue a hostage.",
            "He fights through the compound, dodges gunfire, and escapes.",
        )
        self.assertEqual(genre, "action")
        self.assertEqual(confidence, "medium")
        self.assertGreater(action_score, drama_score)

    def test_classifies_drama_with_multiple_distinct_signals(self):
        genre, confidence, action_score, drama_score = classify_genre(
            "A daughter returns home after her parents' divorce.",
            "She tries to forgive her mother and repair their relationship.",
        )
        self.assertEqual(genre, "drama")
        self.assertEqual(confidence, "medium")
        self.assertGreater(drama_score, action_score)

    def test_two_prompt_cues_are_medium_confidence(self):
        genre, confidence, action_score, drama_score = classify_genre(
            "A soldier rescues someone.",
            "A long story with unrelated family drama.",
        )
        self.assertEqual(genre, "action")
        self.assertEqual(confidence, "medium")
        self.assertEqual((action_score, drama_score), (2, 0))

    def test_keeps_weak_or_mixed_signals_unclear(self):
        genre, confidence, _, _ = classify_genre(
            "A stranger enters a town.",
            "The stranger remembers his childhood and a distant war.",
        )
        self.assertEqual(genre, "unclear")
        self.assertEqual(confidence, "low")

    def test_story_keywords_cannot_assign_a_genre_missing_from_the_prompt(self):
        genre, confidence, action_score, drama_score = classify_genre(
            "A stranger arrives in a quiet town.",
            "The soldier fights through a battle, dodges gunfire, and rescues hostages.",
        )
        self.assertEqual(genre, "unclear")
        self.assertEqual(confidence, "low")
        self.assertEqual((action_score, drama_score), (0, 0))

    def test_extracts_split_from_hugging_face_filename(self):
        self.assertEqual(
            source_split(Path("train-00000-of-00002.parquet")), "train"
        )
        self.assertEqual(
            source_split(Path("validation-00000-of-00001.parquet")), "validation"
        )
        self.assertEqual(source_split(Path("test-00000-of-00001.parquet")), "test")


if __name__ == "__main__":
    unittest.main()
