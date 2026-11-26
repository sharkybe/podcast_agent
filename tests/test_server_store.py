"""The counter store, against memory and (when TEST_DATABASE_URL is set) a real Postgres."""
import datetime as dt
import os
import threading

import pytest

from podcast_skill.server.store import Conversation, MemoryStore

DAY = dt.date(2026, 11, 26)


def _stores():
    yield pytest.param(MemoryStore, id="memory")
    url = os.environ.get("TEST_DATABASE_URL")
    yield pytest.param(url, id="postgres", marks=pytest.mark.skipif(not url, reason="TEST_DATABASE_URL not set"))


@pytest.fixture(params=list(_stores()))
def store(request):
    if request.param is MemoryStore:
        return MemoryStore()
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(request.param, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS usage, conversations, answer_cache")
    from podcast_skill.server.store import PostgresStore

    return PostgresStore(request.param)


def test_take_is_all_or_nothing(store):
    both = [("user", "u", 2), ("ip", "i", 1)]
    assert store.take(DAY, both)
    assert not store.take(DAY, both)            # ip is full, so user is not increased either
    assert store.count(DAY, "user", "u") == 1
    store.give_back(DAY, both)
    assert store.count(DAY, "ip", "i") == 0
    assert store.take(DAY, [("global", "answers", None)])   # no limit


def test_take_under_concurrency_never_exceeds_the_limit(store):
    results = []
    threads = [threading.Thread(target=lambda: results.append(store.take(DAY, [("user", "u", 3)]))) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 3
    assert store.count(DAY, "user", "u") == 3


def test_conversations_and_follow_ups(store):
    store.save_conversation(Conversation("c1", "u", "q", "a", 0), DAY)
    assert store.get_conversation("c1").question == "q"
    assert not store.take_follow_up("c1", "someone-else", 1)
    assert store.take_follow_up("c1", "u", 1)
    assert not store.take_follow_up("c1", "u", 1)
    store.give_back_follow_up("c1")
    assert store.get_conversation("c1").follow_ups == 0


def test_cache_and_cleanup(store):
    store.cache_put("k", {"answer": "שלום", "references": []})
    store.cache_put("k", {"answer": "שלום 2", "references": []})
    assert store.cache_get("k")["answer"] == "שלום 2"
    assert store.cache_get("missing") is None
    store.take(DAY, [("user", "u", 5)])
    store.save_conversation(Conversation("old", "u", "q", "a", 0), DAY)
    store.cleanup(DAY + dt.timedelta(days=1))
    assert store.count(DAY, "user", "u") == 0
    assert store.get_conversation("old") is None
