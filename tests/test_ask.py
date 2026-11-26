import json

import pytest

from podcast_skill.ask import Asker, Turn, format_result, merge_hits, renumber_citations, render_page
from podcast_skill.index import ChunkHit, index_show
from podcast_skill.llm import LLMError, parse_json

from test_index import FakeEmbedder, show, words  # noqa: F401  (fixture)


class FakeLLM:
    """Replies to the expansion and answer prompts with canned JSON; records the prompts."""

    model = "fake-llm"

    def __init__(self, answer=None, queries=("food", "dinner", "eating"), fail=None):
        self.answer = answer if answer is not None else {"answer": "They talk about food [1].", "cited": [1]}
        self.queries = list(queries)
        self.fail = fail or set()     # {"expand", "answer"}
        self.prompts = []

    def generate(self, system, prompt, *, json_output=False, max_output_tokens=2048, thinking_budget=None):
        step = "expand" if "search queries" in system else "answer"
        self.prompts.append((step, prompt))
        if step in self.fail:
            raise LLMError("429 rate limit")
        if step == "expand":
            question = prompt.rsplit("Question: ", 1)[-1]
            return json.dumps({"question": question, "queries": self.queries})
        return json.dumps(self.answer)


@pytest.fixture
def indexed(config, show):  # noqa: F811
    embedder = FakeEmbedder()
    index_show(config, show, embedder=embedder, count_tokens=words)
    return show, embedder


def test_ask_answers_with_references_and_no_transcript_text(config, indexed):
    paths, embedder = indexed
    llm = FakeLLM()
    result = Asker(config, paths, embedder=embedder, llm=llm).ask("what about food?")
    assert not result.fallback
    assert result.answer == "They talk about food [1]."
    assert result.queries == ["food", "dinner", "eating"]
    assert [r.episode_id for r in result.references] == [1]
    assert result.references[0].title == "Episode 1"
    assert result.coverage == "3 of 3 episodes are searchable"
    # transcripts go to the model, never into the result
    answer_prompt = [p for step, p in llm.prompts if step == "answer"][0]
    assert "food food food" in answer_prompt
    assert "food food food" not in json.dumps(result.to_dict(), ensure_ascii=False)


def test_ask_embeds_question_and_queries_in_one_request(config, indexed):
    paths, embedder = indexed
    calls = []
    original = embedder.embed
    embedder.embed = lambda texts: calls.append(list(texts)) or original(texts)
    Asker(config, paths, embedder=embedder, llm=FakeLLM()).ask("sleep?")
    assert calls == [["sleep?", "food", "dinner", "eating"]]


def test_answer_failure_falls_back_to_episode_list(config, indexed):
    paths, embedder = indexed
    result = Asker(config, paths, embedder=embedder, llm=FakeLLM(queries=["school"], fail={"answer"})).ask("school")
    assert result.fallback and result.answer is None
    assert "429" in result.error
    assert result.references[0].episode_id == 3
    assert len({r.episode_id for r in result.references}) == len(result.references)
    assert "No written answer" in format_result(result)


def test_expand_failure_still_answers_with_the_question_alone(config, indexed):
    paths, embedder = indexed
    result = Asker(config, paths, embedder=embedder, llm=FakeLLM(fail={"expand"})).ask("food")
    assert not result.fallback
    assert result.queries == []


def test_follow_up_sends_previous_exchange(config, indexed):
    paths, embedder = indexed
    llm = FakeLLM()
    Asker(config, paths, embedder=embedder, llm=llm).ask("and sleep?", Turn("what about food?", "Food [1]."))
    for _, prompt in llm.prompts:
        assert "Previous question: what about food?" in prompt


def test_not_covered_answer_has_no_references(config, indexed):
    paths, embedder = indexed
    llm = FakeLLM(answer={"answer": "The podcast does not seem to cover it.", "cited": []})
    result = Asker(config, paths, embedder=embedder, llm=llm).ask("cars?")
    assert result.references == []


def test_renumber_citations_drops_unknown_and_renumbers_in_order():
    text, order = renumber_citations("A [3]. B [1, 9]. C [3].", cited=[1, 3, 5], n_passages=5)
    assert text == "A [1]. B [2]. C [1]."
    assert order == [3, 1, 5]   # 5 was cited without a marker: listed last


def test_merge_hits_keeps_best_score_per_chunk():
    a = ChunkHit(1, 1, 0, 0, 10, "x", 0.5)
    b = ChunkHit(1, 1, 0, 0, 10, "x", 0.8)
    c = ChunkHit(2, 1, 1, 10, 20, "y", 0.6)
    merged = merge_hits([[a, c], [b]])
    assert [(h.chunk_id, h.score) for h in merged] == [(1, 0.8), (2, 0.6)]


def test_parse_json_accepts_fenced_reply():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMError):
        parse_json("not json")


def test_render_page_escapes_and_shows_references(config, indexed):
    paths, embedder = indexed
    llm = FakeLLM(answer={"answer": "<b>food</b> [1]", "cited": [1]})
    result = Asker(config, paths, embedder=embedder, llm=llm).ask("food <script>")
    page = render_page([result], "Demo", "fake-llm")
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "&lt;b&gt;food&lt;/b&gt;" in page
    assert "#1 Episode 1" in page
    assert "food food food" not in page


def test_follow_up_after_a_failed_answer_keeps_the_topic(config, indexed):
    paths, embedder = indexed
    asker = Asker(config, paths, embedder=embedder, llm=FakeLLM(fail={"answer"}))
    first = asker.ask("what about food?")
    assert first.fallback
    llm = FakeLLM()
    asker.llm = llm
    asker.ask("and for kids?", Turn.after(first))
    assert "Previous question: what about food?" in llm.prompts[0][1]


def test_build_llm_notes_a_model_outside_free_models(capsys):
    from podcast_skill.llm import build_llm

    config = {"ask": {"llm_model": "gemini-3.5-flash-lite", "free_models": ["gemini-3.5-flash-lite"]}}
    assert build_llm(config).model == "gemini-3.5-flash-lite"
    assert capsys.readouterr().err == ""
    assert build_llm(config, "gemini-4-argon").model == "gemini-4-argon"
    assert "not in ask.free_models" in capsys.readouterr().err
