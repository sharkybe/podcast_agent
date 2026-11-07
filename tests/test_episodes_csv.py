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
