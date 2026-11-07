import json

import pytest

from podcast_skill import discover as d
from podcast_skill import episodes_csv as csvstore
from podcast_skill.episodes_csv import Episode


def test_parse_itunes_results_skips_shows_without_feed(fixtures_dir):
    payload = json.loads((fixtures_dir / "itunes.json").read_text())
    candidates = d.parse_itunes_results(payload)
    assert [c.title for c in candidates] == ["Show A", "Show C"]
    assert candidates[0].feed_url == "https://a.example/rss"
    assert candidates[0].episode_count == 120


def test_parse_feed_orders_oldest_first_and_extracts_fields(fixtures_dir):
    info, episodes = d.parse_feed((fixtures_dir / "feed.xml").read_bytes())
    assert info["title"] == "הפודקאסט לדוגמה"
    assert info["author"] == "Test Author"
    assert [e.id for e in episodes] == [1, 2, 3]
    assert [e.title for e in episodes] == ["Welcome", "Episode 2: Money", "פרק 3 - הורות"]
    assert [e.episode_number for e in episodes] == ["", "2", "3"]
    assert episodes[2].pub_date == "2024-01-03"
    assert episodes[2].duration == "01:02:03"
    assert episodes[2].audio_url == "https://cdn.example/3.mp3"
    assert episodes[2].rss_transcript_url == "https://cdn.example/3.srt"
    assert episodes[0].rss_transcript_url == ""


def test_parse_feed_rejects_non_rss():
    with pytest.raises(ValueError):
        d.parse_feed(b"<html></html>")


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Episode 12: Hi", 12),
        ("ep. 7", 7),
        ("פרק 45 - שלום", 45),
        ("Welcome 2024 special", 2024),
        ("No digits here", None),
    ],
)
def test_extract_episode_number(title, expected):
    assert d.extract_episode_number(title) == expected


def test_match_spotify_episodes_by_title_then_number():
    episodes = [
        Episode(id=1, title="Episode 2: Money!", episode_number="2"),
        Episode(id=2, title="פרק 3 - הורות", episode_number="3"),
        Episode(id=3, title="Unmatched", episode_number=""),
    ]
    spotify_eps = [
        ("episode 2 money", "https://open.spotify.com/episode/aaa"),
        ("פרק 3: הורות (חלק א)", "https://open.spotify.com/episode/bbb"),
    ]
    assert d.match_spotify_episodes(episodes, spotify_eps) == 2
    assert episodes[0].spotify_url.endswith("aaa")
    assert episodes[1].spotify_url.endswith("bbb")  # matched by number
    assert episodes[2].spotify_url == ""


def test_match_by_number_skips_ambiguous():
    episodes = [Episode(id=1, title="Different title", episode_number="5")]
    spotify_eps = [("Part 5 a", "u1"), ("Part 5 b", "u2")]
    assert d.match_spotify_episodes(episodes, spotify_eps) == 0


def test_choose_pick_interactive_and_needs_choice():
    assert d.choose("podcast", ["a", "b"], pick=2, interactive=False) == 1
    with pytest.raises(d.NeedsChoice) as info:
        d.choose("podcast", ["a", "b"], pick=None, interactive=False)
    assert info.value.candidates == ["a", "b"]
    with pytest.raises(ValueError):
        d.choose("podcast", ["a"], pick=3, interactive=False)
    with pytest.raises(ValueError):
        d.choose("podcast", [], pick=None, interactive=True)


def test_resolve_input_name_and_spotify_link(monkeypatch):
    assert d.resolve_input("  My Show ") == ("My Show", None)
    monkeypatch.setattr(d, "fetch_url", lambda url, timeout=30: b'{"title": "Spotify Name"}')
    name, url = d.resolve_input("https://open.spotify.com/show/4XbxH1fXdmJrlrZ0q78kEt?si=abc")
    assert name == "Spotify Name"
    assert url == "https://open.spotify.com/show/4XbxH1fXdmJrlrZ0q78kEt"


def _fake_network(monkeypatch, fixtures_dir):
    def fake_fetch(url, timeout=30):
        if "itunes.apple.com" in url:
            return (fixtures_dir / "itunes.json").read_bytes()
        return (fixtures_dir / "feed.xml").read_bytes()

    monkeypatch.setattr(d, "fetch_url", fake_fetch)


def test_discover_writes_artifacts_and_is_idempotent(config, fixtures_dir, monkeypatch):
    _fake_network(monkeypatch, fixtures_dir)
    paths = d.discover(config, "Show A", pick=1, interactive=False)

    assert paths.episodes_csv.exists()
    show = json.loads(paths.show_json.read_text(encoding="utf-8"))
    assert show["feed_url"] == "https://a.example/rss"
    episodes = csvstore.read_episodes(paths.episodes_csv)
    assert [e.status for e in episodes] == ["pending"] * 3

    # progress survives a re-run
    episodes[0].status = csvstore.UNAVAILABLE
    csvstore.write_episodes(paths.episodes_csv, episodes)
    d.discover(config, "Show A", pick=1, interactive=False)
    again = csvstore.read_episodes(paths.episodes_csv)
    assert [e.status for e in again] == ["unavailable", "pending", "pending"]
    assert [e.id for e in again] == [1, 2, 3]


def test_discover_requires_confirmation_when_not_interactive(config, fixtures_dir, monkeypatch):
    _fake_network(monkeypatch, fixtures_dir)
    with pytest.raises(d.NeedsChoice):
        d.discover(config, "Show A", interactive=False)
