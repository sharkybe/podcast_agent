from podcast_skill import episodes_csv as csvstore
from podcast_skill.episodes_csv import Episode


def test_roundtrip_with_unicode(tmp_path):
    path = tmp_path / "x" / "episodes.csv"
    eps = [Episode(id=2, title="פרק, עם \"פסיק\""), Episode(id=1, title="One", status=csvstore.DONE)]
    csvstore.write_episodes(path, eps)
    back = csvstore.read_episodes(path)
    assert [e.id for e in back] == [1, 2]
    assert back[1].title == 'פרק, עם "פסיק"'
    assert back[0].status == "done"


def test_merge_keeps_progress_and_appends_new_ids():
    existing = [Episode(id=1, title="A", audio_url="a.mp3", status=csvstore.DONE, spotify_url="s1")]
    new = [
        Episode(id=1, title="A renamed", audio_url="a.mp3", pub_date="2024-01-01"),
        Episode(id=2, title="B", audio_url="b.mp3"),
    ]
    merged = csvstore.merge_with_existing(new, existing)
    assert [e.id for e in merged] == [1, 2]
    assert merged[0].status == "done" and merged[0].spotify_url == "s1"
    assert merged[0].pub_date == "2024-01-01"
    assert merged[1].status == "pending"


def _ep(number, date, title=None, **kw):
    return Episode(id=0, title=title or f"פרק {number}", episode_number=str(number) if number else "", pub_date=date, **kw)


def test_assign_ids_uses_the_title_number_and_parks_bonus_episodes_at_the_end():
    eps = [
        _ep(1, "2020-01-01"),
        _ep(2, "2020-01-08"),
        _ep(0, "2020-01-10", title="פרק בונוס"),     # unnumbered, between 2 and 3: nothing is missing here
        _ep(3, "2020-01-15"),
    ]
    csvstore.assign_ids(eps)
    assert [(e.title, e.id) for e in sorted(eps, key=lambda e: e.id)] == [
        ("פרק 1", 1), ("פרק 2", 2), ("פרק 3", 3), ("פרק בונוס", 4),
    ]


def test_assign_ids_unnumbered_intro_takes_the_one_missing_number():
    intro = _ep(0, "2018-07-16", title="הפרק הראשון")
    eps = [intro, _ep(2, "2018-07-31"), _ep(3, "2018-08-05")]
    csvstore.assign_ids(eps)
    assert intro.id == 1


def test_assign_ids_ambiguous_gap_is_not_guessed():
    a, b = _ep(0, "2020-01-02", title="A"), _ep(0, "2020-01-03", title="B")
    eps = [_ep(1, "2020-01-01"), a, b, _ep(4, "2020-01-09")]   # two unnumbered, two free numbers
    csvstore.assign_ids(eps)
    assert (a.id, b.id) == (5, 6)


def test_missing_numbers_become_empty_rows_and_row_index_equals_id(tmp_path):
    path = tmp_path / "episodes.csv"
    eps = [_ep(1, "2020-01-01"), _ep(4, "2020-02-01")]
    csvstore.assign_ids(eps)
    csvstore.write_episodes(path, eps)
    rows = list(__import__("csv").DictReader(path.open(encoding="utf-8")))
    assert [int(r["id"]) for r in rows] == [1, 2, 3, 4]          # row N is episode N
    assert rows[1]["title"] == rows[2]["title"] == "" and rows[1]["status"] == ""
    assert [e.id for e in csvstore.read_episodes(path)] == [1, 4]  # placeholders are not episodes


def test_duplicate_ids_are_refused(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        csvstore.write_episodes(tmp_path / "x.csv", [Episode(id=1, title="a"), Episode(id=1, title="b")])


def test_merge_realigns_shifted_ids_and_resets_only_the_moved_episodes():
    # old CSV: a bonus episode stole id 3, so "פרק 3" sits on id 4
    existing = [
        Episode(id=1, title="פרק 1", episode_number="1", audio_url="1", pub_date="2020-01-01", status=csvstore.DONE, transcript_path="transcripts/001.txt"),
        Episode(id=2, title="פרק 2", episode_number="2", audio_url="2", pub_date="2020-01-08", status=csvstore.DONE, transcript_path="transcripts/002.txt"),
        Episode(id=3, title="בונוס", audio_url="b", pub_date="2020-01-10", status=csvstore.DONE, transcript_path="transcripts/003.txt"),
        Episode(id=4, title="פרק 3", episode_number="3", audio_url="3", pub_date="2020-01-15", status=csvstore.DONE, transcript_path="transcripts/004.txt", spotify_url="s3"),
    ]
    new = [Episode(id=0, title=e.title, episode_number=e.episode_number, audio_url=e.audio_url, pub_date=e.pub_date) for e in existing]
    merged = {e.title: e for e in csvstore.merge_with_existing(new, existing)}
    assert [merged[t].id for t in ("פרק 1", "פרק 2", "פרק 3", "בונוס")] == [1, 2, 3, 4]
    assert merged["פרק 1"].status == "done" and merged["פרק 1"].transcript_path == "transcripts/001.txt"
    assert merged["פרק 3"].status == "pending" and merged["פרק 3"].transcript_path == ""
    assert merged["פרק 3"].spotify_url == "s3"      # the link belongs to the episode, not the row
    assert merged["בונוס"].status == "pending"
