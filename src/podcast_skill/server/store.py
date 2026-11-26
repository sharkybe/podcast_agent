"""Counters, conversations and the answer cache for the hosted service.

Two implementations with the same methods:

- `PostgresStore`: used when `DATABASE_URL` is set (Render's free Postgres). It creates its own
  tables on startup, because a free Render database expires after 30 days and gets recreated.
- `MemoryStore`: for local runs and tests. Everything is lost on restart.

Counters are per day (the caller passes the day, in Israel time). A counter is only increased
if it stays within its limit, and several counters are increased together or not at all, so
two requests at the same moment cannot both take the last question.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence


@dataclass
class Conversation:
    id: str
    user: str
    question: str          # standalone question of the first turn
    answer: str            # its answer ("(no answer was given)" after a fallback)
    follow_ups: int


# (kind, key, limit): kind is "user", "ip" or "global"; limit None = count without a cap
Counter = tuple[str, str, Optional[int]]


class Store(Protocol):
    def take(self, day: dt.date, counters: Sequence[Counter]) -> bool:
        """Add 1 to every counter if none would go over its limit; otherwise change nothing."""
        ...

    def give_back(self, day: dt.date, counters: Sequence[Counter]) -> None:
        """Undo a `take` (the answer failed, so the question doesn't count)."""
        ...

    def count(self, day: dt.date, kind: str, key: str) -> int: ...

    def save_conversation(self, conv: Conversation, day: dt.date) -> None: ...

    def get_conversation(self, conv_id: str) -> Optional[Conversation]: ...

    def take_follow_up(self, conv_id: str, user: str, limit: int) -> bool:
        """Mark one follow-up used, if the conversation is this user's and has one left."""
        ...

    def give_back_follow_up(self, conv_id: str) -> None: ...

    def cache_get(self, key: str) -> Optional[dict[str, Any]]: ...

    def cache_put(self, key: str, value: dict[str, Any]) -> None: ...

    def cleanup(self, before: dt.date) -> None:
        """Drop counters and conversations older than `before`."""
        ...


class MemoryStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[tuple[dt.date, str, str], int] = {}
        self._conversations: dict[str, tuple[Conversation, dt.date]] = {}
        self._cache: dict[str, dict[str, Any]] = {}

    def take(self, day, counters):
        with self._lock:
            for kind, key, limit in counters:
                if limit is not None and self._counts.get((day, kind, key), 0) >= limit:
                    return False
            for kind, key, _ in counters:
                self._counts[(day, kind, key)] = self._counts.get((day, kind, key), 0) + 1
            return True

    def give_back(self, day, counters):
        with self._lock:
            for kind, key, _ in counters:
                if self._counts.get((day, kind, key), 0) > 0:
                    self._counts[(day, kind, key)] -= 1

    def count(self, day, kind, key):
        return self._counts.get((day, kind, key), 0)

    def save_conversation(self, conv, day):
        with self._lock:
            self._conversations[conv.id] = (conv, day)

    def get_conversation(self, conv_id):
        entry = self._conversations.get(conv_id)
        return entry[0] if entry else None

    def take_follow_up(self, conv_id, user, limit):
        with self._lock:
            entry = self._conversations.get(conv_id)
            if not entry or entry[0].user != user or entry[0].follow_ups >= limit:
                return False
            entry[0].follow_ups += 1
            return True

    def give_back_follow_up(self, conv_id):
        with self._lock:
            entry = self._conversations.get(conv_id)
            if entry and entry[0].follow_ups > 0:
                entry[0].follow_ups -= 1

    def cache_get(self, key):
        return self._cache.get(key)

    def cache_put(self, key, value):
        self._cache[key] = value

    def cleanup(self, before):
        with self._lock:
            self._counts = {k: v for k, v in self._counts.items() if k[0] >= before}
            self._conversations = {k: v for k, v in self._conversations.items() if v[1] >= before}


SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    day date NOT NULL,
    kind text NOT NULL,
    key text NOT NULL,
    count integer NOT NULL DEFAULT 0,
    PRIMARY KEY (day, kind, key)
);
CREATE TABLE IF NOT EXISTS conversations (
    id text PRIMARY KEY,
    user_id text NOT NULL,
    day date NOT NULL,
    question text NOT NULL,
    answer text NOT NULL,
    follow_ups integer NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS answer_cache (
    key text PRIMARY KEY,
    value jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
"""


class PostgresStore:
    """One short connection per call: traffic is tiny, and it survives database restarts."""

    def __init__(self, url: str):
        import psycopg

        self._psycopg = psycopg
        self.url = url
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def _connect(self):
        return self._psycopg.connect(self.url, autocommit=False)

    def take(self, day, counters):
        with self._connect() as conn:
            for kind, key, limit in counters:
                conn.execute(
                    "INSERT INTO usage (day, kind, key, count) VALUES (%s, %s, %s, 0) ON CONFLICT DO NOTHING",
                    (day, kind, key),
                )
                row = conn.execute(
                    "UPDATE usage SET count = count + 1 WHERE day = %s AND kind = %s AND key = %s"
                    " AND (%s::int IS NULL OR count < %s::int) RETURNING count",
                    (day, kind, key, limit, limit),
                ).fetchone()
                if row is None:
                    conn.rollback()
                    return False
            conn.commit()
            return True

    def give_back(self, day, counters):
        with self._connect() as conn:
            for kind, key, _ in counters:
                conn.execute(
                    "UPDATE usage SET count = count - 1 WHERE day = %s AND kind = %s AND key = %s AND count > 0",
                    (day, kind, key),
                )

    def count(self, day, kind, key):
        with self._connect() as conn:
            row = conn.execute("SELECT count FROM usage WHERE day = %s AND kind = %s AND key = %s",
                               (day, kind, key)).fetchone()
        return int(row[0]) if row else 0

    def save_conversation(self, conv, day):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, user_id, day, question, answer, follow_ups) VALUES (%s,%s,%s,%s,%s,%s)",
                (conv.id, conv.user, day, conv.question, conv.answer, conv.follow_ups),
            )

    def get_conversation(self, conv_id):
        with self._connect() as conn:
            row = conn.execute("SELECT id, user_id, question, answer, follow_ups FROM conversations WHERE id = %s",
                               (conv_id,)).fetchone()
        return Conversation(*row) if row else None

    def take_follow_up(self, conv_id, user, limit):
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE conversations SET follow_ups = follow_ups + 1"
                " WHERE id = %s AND user_id = %s AND follow_ups < %s RETURNING follow_ups",
                (conv_id, user, limit),
            ).fetchone()
        return row is not None

    def give_back_follow_up(self, conv_id):
        with self._connect() as conn:
            conn.execute("UPDATE conversations SET follow_ups = follow_ups - 1 WHERE id = %s AND follow_ups > 0",
                         (conv_id,))

    def cache_get(self, key):
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM answer_cache WHERE key = %s", (key,)).fetchone()
        if not row:
            return None
        return row[0] if isinstance(row[0], dict) else json.loads(row[0])

    def cache_put(self, key, value):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO answer_cache (key, value) VALUES (%s, %s::jsonb)"
                " ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, created_at = now()",
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def cleanup(self, before):
        with self._connect() as conn:
            conn.execute("DELETE FROM usage WHERE day < %s", (before,))
            conn.execute("DELETE FROM conversations WHERE day < %s", (before,))


def build_store(database_url: Optional[str]) -> Store:
    if database_url:
        # Render shows the URL as postgres://; psycopg accepts both spellings
        return PostgresStore(database_url)
    return MemoryStore()
