#!/usr/bin/env python3
"""Download curated Project Gutenberg plain-text books listed in book_sources.json."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = ROOT / "unused datasets" / "newly dataset" / "books"
SOURCE_FILE = BOOKS_DIR / "book_sources.json"
START_MARKER = b"*** START OF "
END_MARKER = b"*** END OF "


def download_books(books_dir: Path, source_file: Path, force: bool = False) -> None:
    sources = json.loads(source_file.read_text(encoding="utf-8"))
    books = sources.get("books")
    if not isinstance(books, list) or not books:
        raise ValueError(f"No book records found in {source_file}")

    failures: list[str] = []
    downloaded = skipped = 0
    for book in books:
        ebook_id = int(book["id"])
        genre = book["genre"]
        relative_path = Path(book.get("file", f"{genre}/pg-{ebook_id}.txt"))
        path = books_dir / relative_path
        if path.is_file() and not force:
            skipped += 1
            continue

        url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
        request = Request(
            url,
            headers={"User-Agent": "StoryTellingDatasetPrep/1.0 (personal research)"},
        )
        try:
            with urlopen(request, timeout=60) as response:
                content = response.read()
            if START_MARKER not in content or END_MARKER not in content:
                raise ValueError("Downloaded content lacks Gutenberg ebook markers.")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_suffix(path.suffix + ".download")
            temporary_path.write_bytes(content)
            temporary_path.replace(path)
            downloaded += 1
            print(f"Downloaded {ebook_id}: {book['title']} ({len(content):,} bytes)")
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            failures.append(f"{ebook_id} ({book['title']}): {error}")
            print(f"FAILED {ebook_id}: {error}")
        time.sleep(1.0)

    print(f"Downloaded: {downloaded}; already present: {skipped}; failed: {len(failures)}")
    if failures:
        raise RuntimeError("Some Gutenberg downloads failed:\n" + "\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books-dir", type=Path, default=BOOKS_DIR)
    parser.add_argument("--source-file", type=Path, default=SOURCE_FILE)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing source files after validating the downloaded ebook.",
    )
    args = parser.parse_args()
    download_books(args.books_dir, args.source_file, args.force)


if __name__ == "__main__":
    main()
