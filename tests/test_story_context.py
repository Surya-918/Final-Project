import tempfile
import unittest
from pathlib import Path

from src.config.story_memory import (
    CONTINUATION_PATTERN,
    retrieve_memories,
    save_turn,
)
from src.config.test import TextGenerationRequest, build_idea_story_prompt


class StoryContextTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "story-memory.sqlite3"
        save_turn(
            "conversation-1",
            "story-1",
            "A student finds a letter in the campus library.",
            "Maya follows the letter's clue to the old music room.",
            db_path=self.db_path,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_hindi_and_hinglish_followups_are_continuations(self):
        followups = (
            "aage kya hota hai",
            "aage batao",
            "iske baad kya hota hai",
            "phir kya hua",
            "आगे क्या होता है",
            "इसके बाद क्या हुआ",
            "फिर क्या हुआ",
        )
        for followup in followups:
            with self.subTest(followup=followup):
                self.assertIsNotNone(CONTINUATION_PATTERN.search(followup))

    def test_hindi_followup_retrieves_and_passes_story_context(self):
        followup = "aage kya hota hai"
        memories = retrieve_memories(
            "conversation-1",
            followup,
            story_id="story-1",
            db_path=self.db_path,
        )

        self.assertEqual(len(memories), 1)
        prompt = build_idea_story_prompt(
            TextGenerationRequest(prompt=followup, genre="drama"),
            memories,
        )
        self.assertIn("A student finds a letter in the campus library.", prompt)
        self.assertIn(
            "Maya follows the letter's clue to the old music room.",
            prompt,
        )

    def test_unrelated_prompt_does_not_reuse_previous_story(self):
        memories = retrieve_memories(
            "conversation-1",
            "A robot explores Mars",
            story_id="story-1",
            db_path=self.db_path,
        )

        self.assertEqual(memories, [])


if __name__ == "__main__":
    unittest.main()
