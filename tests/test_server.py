import datetime as dt
from zoneinfo import ZoneInfo

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from podcast_skill.ask import Asker  # noqa: E402
from podcast_skill.index import index_show  # noqa: E402
from podcast_skill.paths import ShowPaths  # noqa: E402
from podcast_skill.server.app import (  # noqa: E402
    create_app, embedder_for_index, find_index_file, ip_key, normalise_question,
)
from podcast_skill.server.store import MemoryStore  # noqa: E402

from test_ask import FakeLLM  # noqa: E402
from test_index import FakeEmbedder, show, words  # noqa: E402,F401  (fixture)

TZ = ZoneInfo("Asia/Jerusalem")


class Clock:
    def __init__(self):
        self.now = dt.datetime(2026, 11, 26, 12, 0, tzinfo=TZ)

    def __call__(self):
        return self.now


@pytest.fixture
def setup(config, show):  # noqa: F811
    embedder = FakeEmbedder()
    index_show(config, show, embedder=embedder, count_tokens=words)
    config["server"] = {"questions_per_day": 3, "questions_per_ip_per_day": 5, "answers_per_day": 100}
    llm = FakeLLM()
    asker = Asker(config, show, embedder=embedder, llm=llm)
    store, clock = MemoryStore(), Clock()

    def make(**server):
        config["server"].update(server)
        return TestClient(create_app(config, asker=asker, store=store, now=clock))

    return make, llm, store, clock


def ask(client, question="what about food?", ip="1.2.3.4"):
    return client.post("/api/ask", json={"question": question}, headers={"X-Forwarded-For": ip})


def test_ask_returns_answer_references_and_sets_cookie(setup):
    make, llm, *_ = setup
    client = make()
    r = ask(client)
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "They talk about food [1]."
    assert data["references"][0]["episode_id"] == 1
    assert data["questions_left"] == 2 and data["follow_ups_left"] == 1
    assert data["fallback"] is False and data["cached"] is False
    assert "error" not in data
    assert "food food food" not in r.text          # never transcript text
    assert "pa_uid" in r.cookies


def test_daily_limit_per_user_and_reset_at_midnight_israel_time(setup):
    make, _, _, clock = setup
    client = make()
    for i in range(3):
        assert ask(client, f"question {i}").status_code == 200
    r = ask(client, "one more")
    assert r.status_code == 429 and r.json()["error"] == "daily_limit"
    assert client.get("/api/limits").json()["questions_left"] == 0
    clock.now = dt.datetime(2026, 11, 26, 22, 30, tzinfo=dt.timezone.utc)   # 00:30 in Israel
    assert ask(client, "next day").status_code == 200


def test_ip_cap_applies_across_cookies(setup):
    make, *_ = setup
    for n in range(5):
        client = make()   # a new browser (no cookie) each time, same IP
        assert ask(client, f"q{n}").status_code == 200
    assert ask(make(), "q-last").status_code == 429
    assert ask(make(), "q-other-ip", ip="5.6.7.8").status_code == 200


def test_one_follow_up_per_question(setup):
    make, llm, *_ = setup
    client = make()
    first = ask(client).json()
    r = client.post(f"/api/ask/{first['id']}/follow-up", json={"question": "and kids?"})
    assert r.status_code == 200
    assert r.json()["follow_ups_left"] == 0
    assert r.json()["questions_left"] == 2          # a follow-up is not one of the 10 questions
    assert "Previous question: what about food?" in llm.prompts[-1][1]
    again = client.post(f"/api/ask/{first['id']}/follow-up", json={"question": "and more?"})
    assert again.status_code == 429 and again.json()["error"] == "follow_up_used"


def test_follow_up_needs_the_same_browser(setup):
    make, *_ = setup
    first = ask(make()).json()
    other = make()
    r = other.post(f"/api/ask/{first['id']}/follow-up", json={"question": "and kids?"})
    assert r.status_code == 404


def test_llm_failure_does_not_count(setup):
    make, llm, *_ = setup
    client = make()
    llm.fail = {"answer"}
    data = ask(client).json()
    assert data["fallback"] is True and data["fallback_reason"] == "llm_unavailable"
    assert data["answer"] is None and data["references"]
    assert data["questions_left"] == 3
    # the follow-up stays available after a failed follow-up, too
    llm.fail = set()
    first = ask(client, "something else").json()
    llm.fail = {"answer"}
    r = client.post(f"/api/ask/{first['id']}/follow-up", json={"question": "and kids?"}).json()
    assert r["fallback"] is True and r["follow_ups_left"] == 1


def test_global_budget_gives_episode_list_without_calling_the_llm(setup):
    make, llm, *_ = setup
    client = make(answers_per_day=1)
    assert ask(client, "first").json()["fallback"] is False
    calls = len(llm.prompts)
    data = ask(client, "second").json()
    assert data["fallback"] is True and data["fallback_reason"] == "daily_budget"
    assert data["references"]
    assert len(llm.prompts) == calls                  # no Gemini call
    assert data["questions_left"] == 2               # and it didn't count


def test_repeated_question_is_cached(setup):
    make, llm, *_ = setup
    client = make()
    ask(client, "What about FOOD?")
    calls = len(llm.prompts)
    other = make()
    data = ask(other, "what about food").json()      # another user, same question
    assert data["cached"] is True and data["answer"] == "They talk about food [1]."
    assert len(llm.prompts) == calls
    assert data["questions_left"] == 2               # count_cached: true
    # and a cached answer can still get its follow-up
    assert other.post(f"/api/ask/{data['id']}/follow-up", json={"question": "x"}).status_code == 200


def test_question_validation(setup):
    make, *_ = setup
    client = make()
    assert ask(client, "   ").json()["error"] == "empty_question"
    r = ask(client, "x" * 301)
    assert r.status_code == 400 and r.json()["error"] == "question_too_long"
    assert client.get("/api/limits").json()["questions_left"] == 3


def test_health_and_home(setup):
    make, *_ = setup
    client = make()
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200
    assert 'dir="rtl"' in page.text
    assert "{{" not in page.text                      # every placeholder filled in
    assert "const MAX = 300;" in page.text and "const PER_DAY = 3;" in page.text
    assert "3 of 3 episodes are searchable" in page.text


def test_home_escapes_the_show_title(config, show):  # noqa: F811
    embedder = FakeEmbedder()
    index_show(config, show, embedder=embedder, count_tokens=words)
    asker = Asker(config, show, embedder=embedder, llm=FakeLLM())
    app = create_app(config, asker=asker, store=MemoryStore(), show_title='<b>"x"</b>')
    page = TestClient(app).get("/").text
    assert "<b>" not in page and "&lt;b&gt;" in page


def test_normalise_question_and_ip_key():
    assert normalise_question("  מה  אומרים על אוכל?! ") == normalise_question("מה אומרים על אוכל")
    assert normalise_question("שָׁלוֹם") == "שלום"
    assert ip_key("2001:db8:1:2:aaaa::1", "s") == ip_key("2001:db8:1:2:bbbb::9", "s")
    assert ip_key("1.2.3.4", "s") != ip_key("1.2.3.5", "s")


def test_server_picks_the_compact_index_and_its_dimensions(config, show, tmp_path):  # noqa: F811
    index_show(config, show, embedder=FakeEmbedder(), count_tokens=words)
    big = show.index_file("fake-model")
    small = show.index_dir / "fake-model-4d.db"
    import shutil
    import sqlite3

    shutil.copy(big, small)
    conn = sqlite3.connect(small)
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('dimensions', '4')")
    conn.execute("DELETE FROM chunks WHERE id > 1")
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    assert find_index_file(ShowPaths.at(show.root).root, None) == small
    embedder = embedder_for_index(config, small)
    assert embedder.dimensions == 4
