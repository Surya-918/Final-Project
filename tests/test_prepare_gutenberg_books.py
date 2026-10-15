import re
import tempfile
import unittest
from pathlib import Path

from tools.prepare_gutenberg_books import (
    BOOKS,
    MAX_WORDS,
    chunk_book_text,
    extract_book_text,
    prepare,
    remove_front_matter,
    remove_source_tail,
)


class GutenbergPreparationTests(unittest.TestCase):
    def test_removes_gutenberg_wrapper(self):
        raw = (
            "license text\n*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
            "CHAPTER I.\nA story begins here.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\nfooter\n"
        )
        self.assertEqual(extract_book_text(raw), "CHAPTER I.\nA story begins here.")

    def test_rejects_missing_boundaries(self):
        with self.assertRaises(ValueError):
            extract_book_text("This is not a complete Gutenberg book.")

    def test_removes_embedded_gutenberg_footer_and_edition_promotion(self):
        body = (
            "The story's final sentence ends here. "
            "Did you enjoy this Editor's Cut(tm) edition? "
            "End of Project Gutenberg's Example Book"
        )
        self.assertEqual(
            remove_source_tail(body), "The story's final sentence ends here."
        )

    def test_removes_configured_book_specific_back_matter(self):
        body = "The novel ends here.\n\nAUTHORITIES\nAdditional reference notes."
        self.assertEqual(
            remove_source_tail(body, "AUTHORITIES"), "The novel ends here."
        )

    def test_removes_standalone_transcribers_note_section(self):
        body = "The story ends here.\n\n=Transcriber’s Note=:\nEditorial text."
        self.assertEqual(remove_source_tail(body), "The story ends here.")

    def test_skips_table_of_contents_and_chapter_titles(self):
        body = (
            "CONTENTS\n"
            "Chapter I.\n"
            "Chapter II.\n"
            "CHAPTER I.\n"
            "A long opening paragraph starts the actual story with characters "
            "and events, rather than a table of contents entry.\n"
        )
        self.assertTrue(
            remove_front_matter(body).startswith("A long opening paragraph")
        )

    def test_chunks_are_single_line_and_below_word_limit(self):
        paragraphs = "\n\n".join(
            " ".join(f"word{index}" for index in range(100))
            for _ in range(5)
        )
        chunks = chunk_book_text(paragraphs)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertNotIn("\n", chunk)
            self.assertLessEqual(len(chunk.split()), MAX_WORDS)

    def test_chunker_respects_word_regex_limit_not_only_whitespace_tokens(self):
        chunks = chunk_book_text("alpha/beta " * 150)
        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(len(re.findall(r"\b[\w'-]+\b", chunk)), MAX_WORDS)

    def test_front_matter_removal_skips_contents_and_keeps_short_opening_dialogue(self):
        body = (
            "CONTENTS\n"
            "Chapter I. Contents entry\n"
            "Chapter II. Another contents entry\n\n"
            "CHAPTER I\n"
            '"Tom!"\n\n'
            "A sufficiently long paragraph begins the actual story after that brief "
            "opening line and continues with the character moving through the scene.\n"
        )
        cleaned = remove_front_matter(body)
        self.assertTrue(cleaned.startswith('"Tom!"'))

    def test_chunker_skips_uppercase_chapter_titles(self):
        body = (
            "THE OLD SEA-DOG AT THE ADMIRAL BENBOW\n\n"
            "The opening paragraph introduces the narrator and the old sailor "
            "at the inn beside the sea on a cold and windy evening.\n"
        )
        chunks = chunk_book_text(body)
        self.assertEqual(len(chunks), 1)
        self.assertNotIn("THE OLD SEA-DOG", chunks[0])
        self.assertIn("The opening paragraph", chunks[0])

    def test_chunker_skips_double_dash_chapter_summary_titles(self):
        body = (
            "The Coral Island--Our first cogitations after landing, and the result "
            "of them--We conclude that the island is uninhabited.\n\n"
            "The narrative paragraph introduces the characters and their actions "
            "on the island in a complete scene with enough context for training.\n"
        )
        chunks = chunk_book_text(body)
        self.assertEqual(len(chunks), 1)
        self.assertNotIn("Our first cogitations", chunks[0])
        self.assertIn("The narrative paragraph", chunks[0])

    def test_chunker_skips_decorative_separators(self):
        body = (
            "The final narrative paragraph continues with meaningful context "
            "before the source's decorative divider.\n\n"
            "* * * * *\n"
        )
        chunks = chunk_book_text(body)
        self.assertEqual(len(chunks), 1)
        self.assertNotIn("* * * * *", chunks[0])

    def test_chunker_merges_short_tail_with_next_chunk_when_possible(self):
        short = " ".join(f"short{i}" for i in range(20))
        next_paragraph = " ".join(f"next{i}" for i in range(100))
        chunks = chunk_book_text(f"{short}\n\n{next_paragraph}")
        self.assertEqual(len(chunks), 1)
        self.assertLessEqual(len(chunks[0].split()), MAX_WORDS)
        self.assertIn("short0", chunks[0])
        self.assertIn("next0", chunks[0])

    def test_long_sentence_is_split_into_balanced_chunks(self):
        sentence = " ".join(f"word{index}" for index in range(300))
        chunks = chunk_book_text(sentence)
        self.assertEqual(len(chunks), 2)
        self.assertGreaterEqual(min(len(chunk.split()) for chunk in chunks), 140)
        self.assertLessEqual(max(len(chunk.split()) for chunk in chunks), MAX_WORDS)

    def test_chunker_skips_table_of_contents_dot_leaders(self):
        body = (
            "Chapter I. The Opening\n\n"
            "The chapter starts with a sufficiently long narrative paragraph, "
            "introducing the main character in a real scene by the river.\n\n"
            "From London to Suez, by rail ................. 7 days\n\n"
            "The story continues in a second sufficiently long narrative "
            "paragraph with another character entering the station.\n"
        )
        chunks = chunk_book_text(body)
        self.assertNotIn(".........", " ".join(chunks))
        self.assertNotIn("From London to Suez", " ".join(chunks))

    def test_prepare_keeps_book_sources_in_one_split_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            books_dir = root / "books"
            output_dir = root / "prepared"
            text = (
                "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
                "CHAPTER I.\n\n"
                + "A character entered the room and looked around carefully. " * 8
                + "\n\n"
                + "Another paragraph followed with more narrative prose. " * 8
                + "\n*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
            )
            for book in BOOKS:
                source = books_dir / book["path"]
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(text, encoding="utf-8")

            synthetic_books = tuple(
                {**book, "start_marker": None, "end_marker": None} for book in BOOKS
            )
            manifest = prepare(output_dir, books_dir, synthetic_books)
            train = (output_dir / "gutenberg_train.txt").read_text(encoding="utf-8")
            validation = (output_dir / "gutenberg_validation.txt").read_text(
                encoding="utf-8"
            )

            self.assertIn("<BOS> <action>", train)
            self.assertIn("<BOS> <drama>", train)
            self.assertIn("<BOS> <action>", validation)
            self.assertIn("<BOS> <drama>", validation)
            for book in manifest["books"]:
                self.assertIn(book["split"], ("train", "validation"))
            self.assertTrue((output_dir / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
