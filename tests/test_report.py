import numpy as np
import pytest

from podcast_skill import episodes_csv as csvstore
from podcast_skill.embeddings import normalize
from podcast_skill.index import index_show, search_chunks
from podcast_skill.paths import ShowPaths
from podcast_skill.report import parse_answers, write_report

TOPICS = ["food", "sleep"]


class FakeEmbedder:
    model = "fake"
    dimensions = None

    def embed(self, texts):
        return normalize(np.array([[t.count(x) for x in TOPICS] for t in texts], dtype=np.float32) + 0.01)


@pytest.fixture
def show(config):
    config["index"] = {"chunk_tokens": 4, "overlap_tokens": 0}
    paths = ShowPaths(config, "demo")
    paths.transcripts_dir.mkdir(parents=True)
    paths.transcript_file(1).write_text("[0:00] food food <b>x</b>\n[0:30] sleep sleep", encoding="utf-8")
    csvstore.write_episodes(paths.episodes_csv, [
        csvstore.Episode(id=1, title="Tom & Jerry", pub_date="2020-01-01", status=csvstore.DONE,
                         transcript_path="transcripts/001.txt"),
    ])
    index_show(config, paths, embedder=FakeEmbedder(), count_tokens=lambda t: len(t.split()))
    return paths


def test_search_chunks_returns_k_best_per_query(config, show):
    results = search_chunks(config, show, ["food", "sleep"], k=1, embedder=FakeEmbedder())
    assert [len(r) for r in results] == [1, 1]
    assert "food" in results[0][0].text and "sleep" in results[1][0].text


def test_report_is_escaped_rtl_html_with_scores(config, show):
    out = write_report(config, show, ["food?", "sleep <now>"], top=2, embedder=FakeEmbedder())
    page = out.read_text(encoding="utf-8")
    assert out.parent == show.reports_dir
    assert 'dir="rtl"' in page and "Tom &amp; Jerry" in page
    assert "&lt;b&gt;x&lt;/b&gt;" in page and "<b>x</b>" not in page   # transcript text is escaped
    assert "sleep &lt;now&gt;" in page
    assert page.count("<section>") == 2 and page.count('class="score"') == 4


def test_answers_are_rendered_and_escaped(config, show):
    answers = parse_answers("**bold** first <b>\n\n- one\n- two & three\n===\nsecond answer\n")
    assert len(answers) == 2
    page = write_report(config, show, ["food?", "sleep?"], top=1, embedder=FakeEmbedder(), answers=answers).read_text(encoding="utf-8")
    assert "<strong>bold</strong> first &lt;b&gt;" in page
    assert "<li>one</li><li>two &amp; three</li>" in page
    assert page.count('class="answer"') == 2


def test_report_without_answers_has_no_answer_box(config, show):
    page = write_report(config, show, ["food?"], top=1, embedder=FakeEmbedder()).read_text(encoding="utf-8")
    assert 'class="answer"' not in page


def test_report_needs_questions(config, show):
    with pytest.raises(ValueError):
        search_chunks(config, show, [], embedder=FakeEmbedder())
