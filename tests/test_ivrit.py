from types import SimpleNamespace

from podcast_skill import episodes_csv as csvstore
from podcast_skill import transcribe as t
from podcast_skill.chunking import parse_turns
from podcast_skill.episodes_csv import Episode
from podcast_skill.paths import ShowPaths
from podcast_skill.providers.ivrit import IvritTranscriptProvider, format_timestamp, segments_to_lines


def seg(start, text):
    return SimpleNamespace(start=start, text=text)


class FakeModel:
    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return iter(self.segments), SimpleNamespace(language="he")


def make_provider(config, segments, **kwargs):
    model = FakeModel(segments)
    downloads = []

    def downloader(url, target):
        downloads.append(url)
        target.write_bytes(b"audio")

    provider = IvritTranscriptProvider(config, model_factory=lambda *a, **k: model, downloader=downloader, **kwargs)
    return provider, model, downloads


def test_format_timestamp():
    assert format_timestamp(0) == "0:00"
    assert format_timestamp(75.9) == "1:15"
    assert format_timestamp(3725) == "1:02:05"


def test_segments_become_timestamped_blocks_the_chunker_can_read():
    lines = segments_to_lines([seg(0, " שלום "), seg(10, "מה נשמע"), seg(31, "בסדר"), seg(40, ""), seg(65, "להתראות")])
    assert lines == ["[0:00] שלום", "מה נשמע", "[0:31] בסדר", "[1:05] להתראות"]
    turns = parse_turns("\n".join(lines))
    assert [seconds for seconds, _ in turns] == [0, 31, 65]


def test_fetch_downloads_transcribes_in_hebrew_and_cleans_up(config):
    provider, model, downloads = make_provider(config, [seg(0, "היי"), seg(5, "עולם")])
    lines = provider.fetch(Episode(id=1, title="E1", audio_url="https://x/a.mp3"))
    assert lines == ["[0:00] היי", "עולם"]
    assert downloads == ["https://x/a.mp3"]
    path, kwargs = model.calls[0]
    assert kwargs["language"] == "he"
    assert not __import__("pathlib").Path(path).exists()  # temp audio removed


def test_no_audio_url_or_no_speech_means_no_transcript(config):
    provider, _, downloads = make_provider(config, [])
    assert provider.fetch(Episode(id=1, title="E1")) is None and downloads == []
    assert provider.fetch(Episode(id=2, title="E2", audio_url="https://x/b.mp3")) is None


def test_keep_audio_saves_and_reuses_the_file(config, tmp_path):
    config = {**config, "transcribe": {"ivrit": {"keep_audio": True}}}
    provider, _, downloads = make_provider(config, [seg(0, "היי")], audio_dir=tmp_path / "audio")
    episode = Episode(id=3, title="E3", audio_url="https://x/c.mp3")
    provider.fetch(episode)
    provider.fetch(episode)
    assert (tmp_path / "audio" / "003.mp3").exists()
    assert downloads == ["https://x/c.mp3"]  # second run reuses the file


def test_ivrit_is_only_tried_when_spotify_has_nothing(config):
    paths = ShowPaths(config, "demo")
    csvstore.write_episodes(paths.episodes_csv, [Episode(id=i, title=f"E{i}", audio_url=f"https://x/{i}.mp3") for i in (1, 2)])

    class Spotify:
        name = "spotify"
        open = close = lambda self: None
        fetch = lambda self, episode: ["[0:00] from spotify"] if episode.id == 1 else None

    ivrit, _, downloads = make_provider(config, [seg(0, "local")])
    counts = t.transcribe(config, paths, providers=[Spotify(), ivrit])
    assert counts["done"] == 2
    assert downloads == ["https://x/2.mp3"]  # episode 1 never reached the local model
    assert paths.transcript_file(1).read_text(encoding="utf-8") == "[0:00] from spotify"
    assert paths.transcript_file(2).read_text(encoding="utf-8") == "[0:00] local"


def test_build_providers_knows_ivrit(config):
    providers = t.build_providers(config, names=["ivrit"])
    assert [p.name for p in providers] == ["ivrit"]
