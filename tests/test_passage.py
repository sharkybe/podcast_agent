import pytest

from podcast_skill import episodes_csv as csvstore
from podcast_skill.index import status_report
from podcast_skill.passage import get_passage, parse_time
from podcast_skill.paths import ShowPaths


@pytest.fixture
def show(config):
    paths = ShowPaths(config, "demo")
    paths.transcripts_dir.mkdir(parents=True)
    paths.transcript_file(1).write_text(
        "[0:05] first turn\nstill first\n[1:00] second\n[2:30] third\n[10:00] far away", encoding="utf-8"
    )
    csvstore.write_episodes(paths.episodes_csv, [
        csvstore.Episode(id=1, title="One", status=csvstore.DONE, transcript_path="transcripts/001.txt"),
        csvstore.Episode(id=2, title="Two", status=csvstore.UNAVAILABLE),
    ])
    return paths


def test_parse_time_formats():
    assert parse_time("90") == 90
    assert parse_time("1:30") == 90
    assert parse_time("1:02:03") == 3723
    with pytest.raises(ValueError):
        parse_time("soon")


def test_passage_returns_turns_in_range_plus_the_one_running_at_the_start(show):
    text = get_passage(show, 1, parse_time("1:30"), parse_time("3:00"))
    assert text == "[1:00] second\n[2:30] third"   # [1:00] began before 1:30 and may still be running


def test_passage_default_span_and_multi_line_turns(show):
    assert get_passage(show, 1, 0) == "[0:05] first turn\nstill first\n[1:00] second\n[2:30] third"


def test_passage_errors(show):
    with pytest.raises(FileNotFoundError, match="no transcript"):
        get_passage(show, 2, 0)
    with pytest.raises(ValueError, match="No episode"):
        get_passage(show, 99, 0)
    with pytest.raises(ValueError, match="before"):
        get_passage(show, 1, 100, 50)


def test_status_report(show):
    report = status_report(show)
    assert "Episodes: 2" in report
    assert "done: 1" in report
    assert "unavailable: 1  (ids: 2)" in report
    assert "Index: none yet" in report
