import numpy as np
import pytest

from podcast_skill import episodes_csv as csvstore
from podcast_skill.embeddings import OpenAIEmbedder, normalize
from podcast_skill.index import format_hits, index_show, search_episodes
from podcast_skill.paths import ShowPaths

TOPICS = ["food", "sleep", "school"]


class FakeEmbedder:
    """Embeds a text as the count of topic keywords it contains (one axis per topic)."""

    def __init__(self, model="fake-model", dimensions=None):
        self.model = model
        self.dimensions = dimensions
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        vectors = np.array([[text.count(topic) for topic in TOPICS] for text in texts], dtype=np.float32)
        vectors += 0.01  # no all-zero vectors
        return normalize(vectors)


def words(text):
    return len(text.split())


EPISODES = {
    1: "[0:00] food food food\n[0:30] food dinner\n[1:00] sleep",
    2: "[0:00] sleep sleep sleep\n[0:30] night sleep",
    3: "[0:00] school school\n[0:30] school bus",
}


@pytest.fixture
def show(config, tmp_path):
    config["index"] = {"chunk_tokens": 4, "overlap_tokens": 0}
    paths = ShowPaths(config, "demo")
    paths.transcripts_dir.mkdir(parents=True)
    episodes = []
    for episode_id, text in EPISODES.items():
        paths.transcript_file(episode_id).write_text(text, encoding="utf-8")
        episodes.append(csvstore.Episode(
            id=episode_id, title=f"Episode {episode_id}", pub_date=f"2020-01-0{episode_id}",
            status=csvstore.DONE, transcript_path=f"transcripts/{episode_id:03d}.txt",
        ))
    csvstore.write_episodes(paths.episodes_csv, episodes)
    return paths


def run_index(config, paths, embedder, **kwargs):
    return index_show(config, paths, embedder=embedder, count_tokens=words, **kwargs)


def test_index_then_search_ranks_episodes_by_topic(config, show):
    embedder = FakeEmbedder()
    counts = run_index(config, show, embedder)
    assert counts["indexed"] == 3 and counts["chunks"] > 0

    results = search_episodes(config, show, "sleep", top_chunks=4, embedder=embedder)
    assert results[0].episode_id == 2
    assert results[0].title == "Episode 2"
    assert results[0].pub_date == "2020-01-02"
    assert [p.t_start for p in results[0].passages] == sorted(p.t_start for p in results[0].passages)


def test_search_can_be_limited_to_episodes(config, show):
    embedder = FakeEmbedder()
    run_index(config, show, embedder)
    results = search_episodes(config, show, "food", top_chunks=10, episode_ids=[2, 3], embedder=embedder)
    assert {r.episode_id for r in results} <= {2, 3}


def test_min_score_drops_weak_passages(config, show):
    embedder = FakeEmbedder()
    run_index(config, show, embedder)
    results = search_episodes(config, show, "school", top_chunks=10, min_score=0.9, embedder=embedder)
    assert [r.episode_id for r in results] == [3]


def test_second_run_embeds_nothing(config, show):
    embedder = FakeEmbedder()
    run_index(config, show, embedder)
    calls = embedder.calls
    counts = run_index(config, show, embedder)
    assert counts["indexed"] == 0 and counts["unchanged"] == 3
    assert embedder.calls == calls


def test_changed_transcript_is_reindexed_alone(config, show):
    embedder = FakeEmbedder()
    run_index(config, show, embedder)
    show.transcript_file(2).write_text("[0:00] sleep again", encoding="utf-8")
    counts = run_index(config, show, embedder)
    assert counts["indexed"] == 1 and counts["unchanged"] == 2
    results = search_episodes(config, show, "sleep", top_chunks=50, episode_ids=[2], embedder=embedder)
    assert [p.text for p in results[0].passages] == ["sleep again"]


def test_changed_chunk_settings_reindex_everything(config, show):
    embedder = FakeEmbedder()
    run_index(config, show, embedder)
    config["index"]["chunk_tokens"] = 6
    assert run_index(config, show, embedder)["indexed"] == 3


def test_episode_no_longer_done_is_removed(config, show):
    embedder = FakeEmbedder()
    run_index(config, show, embedder)
    episodes = csvstore.read_episodes(show.episodes_csv)
    episodes[0].status = csvstore.UNAVAILABLE
    csvstore.write_episodes(show.episodes_csv, episodes)
    assert run_index(config, show, embedder)["removed"] == 1
    results = search_episodes(config, show, "food", top_chunks=50, embedder=embedder)
    assert 1 not in {r.episode_id for r in results}


def test_each_model_gets_its_own_index_file(config, show):
    large, small = FakeEmbedder("big"), FakeEmbedder("small")
    run_index(config, show, large)
    assert not show.index_file("small").exists()
    run_index(config, show, small)
    assert show.index_file("big").exists() and show.index_file("small").exists()
    assert show.index_file("big", 1024).name == "big-1024d.db"


def test_rebuild_embeds_everything_again(config, show):
    embedder = FakeEmbedder()
    run_index(config, show, embedder)
    assert run_index(config, show, embedder, rebuild=True)["indexed"] == 3


def test_search_before_indexing_explains_what_to_do(config, show):
    with pytest.raises(FileNotFoundError, match="run the index step"):
        search_episodes(config, show, "food", embedder=FakeEmbedder())


def test_empty_query_rejected(config, show):
    with pytest.raises(ValueError):
        search_episodes(config, show, "  ", embedder=FakeEmbedder())


def test_format_hits(config, show):
    embedder = FakeEmbedder()
    run_index(config, show, embedder)
    text = format_hits(search_episodes(config, show, "sleep", top_chunks=3, embedder=embedder))
    assert "#2 Episode 2" in text and "[0:00]" in text
    assert format_hits([]) == "No matching passages."


class _FakeOpenAI:
    """Mimics client.embeddings.create, returning items out of order to test sorting."""

    def __init__(self):
        self.requests = []
        self.embeddings = self

    def create(self, input, model, **kwargs):
        self.requests.append((list(input), model, kwargs))
        items = [type("Item", (), {"index": i, "embedding": [float(len(t)), 1.0]})() for i, t in enumerate(input)]
        return type("Response", (), {"data": list(reversed(items))})()


def test_openai_embedder_batches_orders_and_normalizes():
    client = _FakeOpenAI()
    embedder = OpenAIEmbedder("m", dimensions=2, batch_size=2, client=client)
    vectors = embedder.embed(["a", "bbb", "cc"])
    assert [len(r[0]) for r in client.requests] == [2, 1]
    assert client.requests[0][2] == {"dimensions": 2}
    assert vectors.shape == (3, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert vectors[1][0] > vectors[0][0]  # "bbb" (3) kept its place after "a" (1)


def test_openai_embedder_requires_api_key(monkeypatch):
    monkeypatch.delenv("PODCAST_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PODCAST_TEST_KEY"):
        OpenAIEmbedder("m", api_key_env="PODCAST_TEST_KEY").embed(["x"])
