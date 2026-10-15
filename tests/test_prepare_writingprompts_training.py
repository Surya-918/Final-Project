import unittest
from pathlib import Path

from tools.prepare_writingprompts_training import (
    clean_text,
    contains_abusive_language,
    contains_minor_sexual_content,
    contains_post_noise,
    format_example,
    is_non_story_prompt,
    load_tokenizer,
    normalized_prompt_key,
    normalized_story_key,
    serialize_example,
)


class WritingPromptsTrainingPreparationTests(unittest.TestCase):
    def test_rejects_profanity_direct_abuse_and_sexual_violence(self):
        self.assertTrue(contains_abusive_language("The character said " + "f" + "uck."))
        self.assertTrue(contains_abusive_language("He called her an idiot."))
        self.assertTrue(contains_abusive_language("The report describes rape."))
        self.assertTrue(contains_abusive_language("That was a mother fucker of a day."))
        self.assertFalse(contains_abusive_language("The foolish choice was corrected."))
        self.assertFalse(contains_abusive_language("The family returned home safely."))

    def test_removes_writing_prompts_prefix_and_normalizes_whitespace(self):
        self.assertEqual(
            clean_text("[ WP ] A premise\n with spaces", is_prompt=True),
            "A premise with spaces",
        )

    def test_rejects_replacement_and_control_characters(self):
        self.assertIsNone(clean_text("broken \ufffd text"))
        self.assertIsNone(clean_text("broken\x00text"))

    def test_rejects_off_topic_and_workshop_prompts(self):
        self.assertTrue(is_non_story_prompt("[ OT ] Writing Workshop #49: Conflict"))
        self.assertTrue(is_non_story_prompt("Writing challenge: write a character"))
        self.assertFalse(is_non_story_prompt("[WP] A family reunites after years apart"))

    def test_rejects_author_feedback_and_editorial_tails(self):
        self.assertTrue(
            contains_post_noise("(Would love feedback, forgive my English, sorry if it sucks.)")
        )
        self.assertTrue(
            contains_post_noise("-- I apologize for the utter destruction above as I attempted a poem.")
        )
        self.assertTrue(contains_post_noise("EDIT: Wrote this as an opening scene, not a full story."))
        self.assertTrue(contains_post_noise("My ending got lazy due to time constraints."))
        self.assertFalse(contains_post_noise("The couple apologized and began again."))

    def test_flags_explicit_sexual_content_near_minor_references(self):
        self.assertTrue(
            contains_minor_sexual_content(
                "A child was mentioned in a story containing explicit sexual content."
            )
        )
        self.assertTrue(
            contains_minor_sexual_content(
                "A little girl asks why they fight; an adult calls her mother a whore and explains that her father would fuck."
            )
        )
        self.assertFalse(
            contains_minor_sexual_content(
                "A child developed a medical condition after a fall."
            )
        )
        self.assertFalse(
            contains_minor_sexual_content(
                "A child heard an adult yell fuck after dropping a plate."
            )
        )
        self.assertTrue(
            contains_minor_sexual_content(
                "A boy asks a girl out; it is high school.",
                "It is mating season; the narrator uses an aphrodisiac.",
            )
        )

    def test_formats_to_the_app_idea_story_template(self):
        self.assertEqual(
            format_example("action", "A rescue mission begins.", "They escape."),
            "<BOS> <action> Idea: A rescue mission begins.\n"
            "Story: They escape. <EOS>",
        )

    def test_serializes_logical_newline_without_splitting_training_row(self):
        self.assertEqual(
            serialize_example("drama", "A family reunites.", "They forgive."),
            "<BOS> <drama> Idea: A family reunites.\\n"
            "Story: They forgive. <EOS>",
        )

    def test_rejects_unsupported_genre(self):
        with self.assertRaises(ValueError):
            format_example("fantasy", "A premise.", "A story.")

    def test_prompt_deduplication_ignores_case_and_punctuation(self):
        self.assertEqual(
            normalized_prompt_key("[WP] Find the hidden key!"),
            normalized_prompt_key("find the hidden key"),
        )

    def test_story_deduplication_ignores_case_and_punctuation(self):
        self.assertEqual(
            normalized_story_key("The same ending, again!"),
            normalized_story_key("the same ending again"),
        )

    def test_tokenizer_does_not_silently_truncate_long_examples(self):
        tokenizer_path = (
            Path(__file__).resolve().parents[1]
            / "trained model-college"
            / "tokenizer.json"
        )
        tokenizer = load_tokenizer(tokenizer_path)
        long_text = " ".join(["untruncated"] * 2000)
        self.assertGreater(
            len(tokenizer.encode(long_text, add_special_tokens=False).ids), 512
        )


if __name__ == "__main__":
    unittest.main()
