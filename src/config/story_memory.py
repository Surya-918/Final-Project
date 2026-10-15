import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
configured_db_path = Path(os.getenv("STORY_MEMORY_DB", ".story_memory.sqlite3"))
MEMORY_DB_PATH = (
    configured_db_path
    if configured_db_path.is_absolute()
    else PROJECT_ROOT / configured_db_path
)
STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been",
    "before", "being", "but", "can", "could", "did", "does", "for", "from",
    "had", "has", "have", "her", "here", "him", "his", "how", "into", "its",
    "just", "more", "not", "our", "out", "she", "should", "some", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "too", "was", "were", "what", "when", "where", "which",
    "who", "will", "with", "would", "you", "your", "to", "while",
    "appear", "appears", "kill", "kills", "meanwhile", "save", "saves",
    "tries", "try",
}
CONTINUATION_PATTERN = re.compile(
    r"\b(?:continue|continuing|keep going|go on|carry on|next scene|"
    r"what happens next|what happened next|same story|after that|then what|"
    r"aage(?:\s+(?:batao|badhao|chalao|likho|kya(?:\s+(?:hota|hua|hoga)\s+hai)?))?|"
    r"iske\s+baad|phir\s+kya(?:\s+(?:hua|hota|hoga))?)\b|"
    r"आगे(?:\s+(?:बताओ|बढ़ाओ|चलाओ|लिखो|क्या(?:\s+(?:होता|हुआ|होगा)\s+है)?))?|"
    r"इसके\s+बाद|फिर\s+क्या(?:\s+(?:हुआ|होता|होगा))?",
    re.IGNORECASE,
)
MIN_MEMORY_TERM_OVERLAP = 3
MIN_MEMORY_QUERY_COVERAGE = 0.5


def _connect(db_path=MEMORY_DB_PATH):
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS story_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            story_id TEXT NOT NULL DEFAULT '',
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(story_turns)")
    }
    if "story_id" not in columns:
        connection.execute(
            "ALTER TABLE story_turns ADD COLUMN story_id TEXT NOT NULL DEFAULT ''"
        )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS story_turns_story "
        "ON story_turns (conversation_id, story_id, id)"
    )
    return connection


def save_turn(conversation_id, story_id, prompt, response, db_path=MEMORY_DB_PATH):
    if not conversation_id:
        return
    with closing(_connect(db_path)) as connection:
        connection.execute(
            "INSERT INTO story_turns (conversation_id, story_id, prompt, response) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, story_id, prompt, response),
        )
        connection.commit()


def retrieve_memories(
    conversation_id, query, story_id="", limit=4, db_path=MEMORY_DB_PATH
):
    if not conversation_id:
        return []
    with closing(_connect(db_path)) as connection:
        if story_id:
            rows = connection.execute(
                "SELECT id, story_id, prompt, response FROM story_turns "
                "WHERE conversation_id = ? AND story_id = ? ORDER BY id",
                (conversation_id, story_id),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT id, story_id, prompt, response FROM story_turns "
                "WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()

    if not rows:
        return []
    if not story_id:
        active_story_id = rows[-1][1]
        rows = [row for row in rows if row[1] == active_story_id]

    query_terms = {
        term for term in re.findall(r"[a-z0-9]{2,}", query.lower())
        if term not in STOP_WORDS
    }
    continuing = bool(CONTINUATION_PATTERN.search(query))
    ranked = []
    for row in rows:
        content_terms = set(
            re.findall(r"[a-z0-9]{2,}", f"{row[2]} {row[3]}".lower())
        ) - STOP_WORDS
        overlap = len(query_terms & content_terms)
        score = overlap / ((len(query_terms) * len(content_terms)) ** 0.5) if overlap else 0
        query_coverage = overlap / max(1, len(query_terms))
        ranked.append((score, overlap, query_coverage, row[0], row))

    if not continuing and not any(
        overlap >= MIN_MEMORY_TERM_OVERLAP
        and query_coverage >= MIN_MEMORY_QUERY_COVERAGE
        for _, overlap, query_coverage, _, _ in ranked
    ):
        return []

    ranked.sort(key=lambda item: (item[0], item[3]), reverse=True)
    if continuing:
        selected = {row[0]: row for row in rows[-max(1, limit):]}
    else:
        selected = {
            row[0]: row
            for _, overlap, query_coverage, _, row in ranked
            if overlap >= MIN_MEMORY_TERM_OVERLAP
            and query_coverage >= MIN_MEMORY_QUERY_COVERAGE
        }
        selected = dict(
            sorted(selected.items())[-max(1, limit):]
        )

    return [
        {"story_id": row[1], "prompt": row[2], "response": row[3]}
        for row in sorted(selected.values(), key=lambda item: item[0])
    ]