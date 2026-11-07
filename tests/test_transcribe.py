import pytest

from podcast_skill import episodes_csv as csvstore
from podcast_skill import transcribe as t
from podcast_skill.episodes_csv import Episode
from podcast_skill.paths import ShowPaths, slugify
from podcast_skill.providers.spotify import clean_transcript_lines


class FakeProvider:
    name = "fake"

    def __init__(self, transcripts, fail_on=()):
        self.transcripts = transcripts
        self.fail_on = set(fail_on)
        self.opened = self.closed = False
        self.calls = []

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def fetch(self, episode):
        self.calls.append(episode.id)
        if episode.id in self.fail_on:
            raise RuntimeError("boom")
        return self.transcripts.get(episode.id)


@pytest.fixture
def show(config):
    paths = ShowPaths(config, "demo")
    csvstore.write_episodes(paths.episodes_csv, [Episode(id=i, title=f"E{i}") for i in (1, 2, 3, 4)])
    return paths


def test_marks_done_unavailable_failed(config, show):
    provider = FakeProvider({1: ["hello", "world"], 2: None}, fail_on={3})
    counts = t.transcribe(config, show, providers=[provider])
    assert counts == {"done": 1, "unavailable": 2, "failed": 1}  # id 4 has no transcript
    assert provider.opened and provider.closed

    by_id = {e.id: e for e in csvstore.read_episodes(show.episodes_csv)}
    assert by_id[1].status == "done"
    assert by_id[1].transcript_path == "transcripts/001.txt"
    assert show.transcript_file(1).read_text(encoding="utf-8") == "hello\nworld"
    assert by_id[2].status == "unavailable"
    assert by_id[3].status == "failed"
    assert not show.transcript_file(2).exists()


def test_rerun_skips_done_and_unavailable_but_retries_failed(config, show):
    t.transcribe(config, show, providers=[FakeProvider({1: ["x"]}, fail_on={3})])
    second = FakeProvider({3: ["now works"]})
    t.transcribe(config, show, providers=[second])
    assert second.calls == [3]
    by_id = {e.id: e for e in csvstore.read_episodes(show.episodes_csv)}
    assert by_id[3].status == "done"
    assert by_id[2].status == "unavailable"


def test_retry_unavailable_and_limit(config, show):
    t.transcribe(config, show, providers=[FakeProvider({})])
    assert {e.status for e in csvstore.read_episodes(show.episodes_csv)} == {"unavailable"}
    retry = FakeProvider({1: ["a"], 2: ["b"]})
    t.transcribe(config, show, providers=[retry], retry_unavailable=True, limit=2)
    assert retry.calls == [1, 2]
    statuses = {e.id: e.status for e in csvstore.read_episodes(show.episodes_csv)}
    assert statuses == {1: "done", 2: "done", 3: "unavailable", 4: "unavailable"}


def test_second_provider_used_when_first_has_nothing(config, show):
    first, second = FakeProvider({}), FakeProvider({1: ["from second"]})
    second.name = "second"
    t.transcribe(config, show, providers=[first, second], limit=1)
    assert show.transcript_file(1).read_text(encoding="utf-8") == "from second"


def test_missing_csv_raises(config):
    with pytest.raises(FileNotFoundError):
        t.transcribe(config, ShowPaths(config, "nope"), providers=[FakeProvider({})])


def test_unknown_provider_name(config):
    config["transcribe"]["providers"] = ["whisper"]
    with pytest.raises(ValueError):
        t.build_providers(config)


def test_clean_transcript_lines():
    raw = [
        "  hello ",
        "",
        "This transcript was generated automatically. Its accuracy may vary.",
        "3:07",
        "hello",
        "דובר או דוברת 1",
        "world: not a timestamp",
        "world",
    ]
    # repeated lines are kept: "כן." said twice is two lines of speech
    assert clean_transcript_lines(raw) == ["hello", "hello", "world: not a timestamp", "world"]


def test_slugify_keeps_hebrew_and_strips_punctuation():
    assert slugify("The Show: Part 1!") == "the-show-part-1"
    assert slugify("הפודקאסט  לדוגמה") == "הפודקאסט-לדוגמה"
    assert slugify("!!!") == "show"


def test_only_ids_forces_rerun_regardless_of_status(config, show):
    t.transcribe(config, show, providers=[FakeProvider({})])  # everything -> unavailable
    forced = FakeProvider({2: ["found now"]})
    t.transcribe(config, show, providers=[forced], only_ids=[2])
    assert forced.calls == [2]
    by_id = {e.id: e for e in csvstore.read_episodes(show.episodes_csv)}
    assert by_id[2].status == "done" and by_id[1].status == "unavailable"


def test_format_transcript_rows_adds_timestamp_to_first_line_of_each_turn():
    from podcast_skill.providers.spotify import format_transcript_rows

    rows = [
        "This transcript was generated automatically. Its accuracy may vary.",
        "0:00\nדובר או דוברת מס {0}",   # header row: timestamp + speaker label
        "first line",
        "",                              # not rendered yet / empty placeholder
        "second line of same turn",
        "1:05\nדובר או דוברת מס {0}",
        "new turn",
        "1:07:30\nדובר או דוברת מס {0}",  # hour-long episodes use h:mm:ss
        "late turn",
    ]
    assert format_transcript_rows(rows) == [
        "[0:00] first line",
        "second line of same turn",
        "[1:05] new turn",
        "[1:07:30] late turn",
    ]
    assert format_transcript_rows(["plain", "text"]) == ["plain", "text"]  # no headers: no timestamps
