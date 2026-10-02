"""Long-Term Memory (LTM): facts that survive across sessions.

LTM is a separate table, `long_term_memory`, in agent.db. Each row is one
fact, identified by (category, scope, mem_key):

    category   'document_fact' or 'preference'
    scope      the PDF filename for document facts, 'user' for preferences
    mem_key    e.g. 'cooling tower fan | motor horsepower'
    value      e.g. '30 HP'

That unique key is what prevents duplicates: saving the same fact again is a
no-op, and saving a new value for the same key updates the old row.
"""
import re
import sqlite3
from datetime import datetime

from memory_utils import keywords, normalize_space

LTM_DB_PATH = "agent.db"
LTM_SEARCH_LIMIT = 5
MAX_PREFERENCES_IN_PROMPT = 3
MAX_VALUE_LENGTH = 300

# The user explicitly asks the agent to keep something for later
REMEMBER_PATTERN = re.compile(
    r"\b(remember|note that|keep in mind|save this|for future|from now on|i prefer)\b",
    re.IGNORECASE,
)
# The user asks about an earlier session, so STM alone can't be enough
PAST_SESSION_PATTERN = re.compile(
    r"\b(last time|last session|previous session|previously|you told me|i told you|"
    r"do you remember|in the past)\b",
    re.IGNORECASE,
)


def wants_to_remember(query):
    return bool(REMEMBER_PATTERN.search(query))


def asks_about_past_sessions(query):
    return bool(PAST_SESSION_PATTERN.search(query))


class LongTermMemory:
    def __init__(self, db_path=LTM_DB_PATH):
        self.db_path = db_path
        self.setup()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def setup(self):
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    mem_key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    UNIQUE (category, scope, mem_key)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def save(self, category, scope, key, value, source):
        """Stores one fact. Returns (status, old_value).

        status is 'inserted', 'updated' (old_value holds the replaced value),
        'duplicate' (same fact already stored) or 'skipped' (empty fact).
        """
        key = normalize_space(key)
        value = " ".join(str(value or "").split())[:MAX_VALUE_LENGTH]
        if not key or not value:
            return "skipped", None

        now = datetime.now().isoformat(timespec="seconds")
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, value FROM long_term_memory "
                "WHERE category = ? AND scope = ? AND mem_key = ?",
                (category, scope, key),
            ).fetchone()

            if row and row["value"].lower() == value.lower():
                return "duplicate", None

            if row:
                conn.execute(
                    "UPDATE long_term_memory SET value = ?, source = ?, updated_at = ? "
                    "WHERE id = ?",
                    (value, source, now, row["id"]),
                )
                conn.commit()
                return "updated", row["value"]

            conn.execute(
                "INSERT INTO long_term_memory "
                "(category, scope, mem_key, value, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (category, scope, key, value, source, now, now),
            )
            conn.commit()
            return "inserted", None
        finally:
            conn.close()

    def search(self, query, filename=None, limit=LTM_SEARCH_LIMIT):
        """Returns the stored facts most related to the query.

        Scoring is simple keyword overlap between the query and the fact.
        Facts from the PDF currently open get a small boost. A few user
        preferences are always included, because they apply to every answer.
        """
        query_words = keywords(query)
        conn = self._connect()
        try:
            rows = [dict(r) for r in conn.execute("SELECT * FROM long_term_memory")]
        finally:
            conn.close()

        facts, preferences = [], []
        for row in rows:
            if row["category"] == "preference":
                preferences.append(row)
                continue
            fact_words = keywords(f"{row['mem_key']} {row['value']} {row['scope']}")
            score = len(query_words & fact_words)
            if score == 0:
                continue
            if filename and row["scope"] == filename:
                score += 0.5
            row["score"] = score
            facts.append(row)

        facts.sort(key=lambda r: r["score"], reverse=True)
        preferences.sort(key=lambda r: r["updated_at"], reverse=True)
        return facts[:limit] + preferences[:MAX_PREFERENCES_IN_PROMPT]

    def all(self):
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM long_term_memory ORDER BY category, scope, mem_key"
            )
            return [dict(r) for r in rows]
        finally:
            conn.close()


def format_records(records):
    """Formats LTM rows for the prompt, each labeled [LTM-id]"""
    if not records:
        return "(nothing relevant from earlier sessions)"
    lines = []
    for r in records:
        if r["category"] == "preference":
            lines.append(f"[LTM-{r['id']}] User preference: {r['value']}")
        else:
            lines.append(
                f"[LTM-{r['id']}] From {r['scope']} (saved {r['updated_at'][:10]}): "
                f"{r['mem_key']} = {r['value']}"
            )
    return "\n".join(lines)
