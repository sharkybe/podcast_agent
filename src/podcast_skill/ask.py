"""Free answer path: answer a question without Claude, with a fixed pipeline.

    expand (LLM) -> embed (OpenAI) -> search and merge -> answer (LLM) -> references

It mirrors what SKILL.md step 4 asks Claude to do, in at most two LLM calls:

1. The LLM rewrites the question into a few Hebrew search queries (and, for a follow-up, into a
   question that stands on its own).
2. The original question and the queries are embedded in one request, with the same model as
   the index.
3. Each vector is searched; hits are merged by chunk (best score wins) and the best chunks,
   with the chunk that follows each one, become the context.
4. The LLM answers from that context only, in its own words, and says which passages it used.
5. The result is the answer plus references (episode, date, time). Transcript text is never
   part of the result: it only goes to the LLM.

If the LLM fails (free-tier rate limit, network), the result has no answer and lists the best
matching episodes instead (`fallback=True`).
"""
from __future__ import annotations

import datetime as dt
import html
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from podcast_skill import episodes_csv as csvstore
from podcast_skill.chunking import format_time
from podcast_skill.config import section
from podcast_skill.embeddings import Embedder, build_embedder
from podcast_skill.index import ChunkHit, VectorIndex
from podcast_skill.llm import LLM, LLMError, build_llm, parse_json
from podcast_skill.paths import ShowPaths

EXPAND_SYSTEM = """\
You write search queries for a semantic search over automatic transcripts of a Hebrew podcast.

Return JSON: {{"question": "...", "queries": ["...", ...]}}
- "question": the user's question, rewritten so it stands on its own (resolve "it", "that episode"
  and the like from the previous exchange, if one is given). Keep the user's language.
- "queries": exactly {n} short search queries in Hebrew. Use different wordings and synonyms of the
  topic, the way people say it in spoken Hebrew, not a word-for-word translation of the question.
  For example, for eating disorders: "הפרעות אכילה", "אכילה רגשית", "דיאטה וגוף".
Do not answer the question."""

ANSWER_SYSTEM = """\
You are the woman who hosts and records a Hebrew parenting podcast, answering a listener's question.
You answer from numbered passages of the podcast's automatic transcripts: they are what was said on
your show. Each passage header gives the episode number, title, date and start time.

Return JSON: {"answer": "...", "cited": [numbers of the passages you used]}

Rules for "answer":
- Answer in the language of the question.
- Speak as the host, in the first person singular and in the feminine form, warmly and directly to
  the listener: "בפרק 72, 'התמודדות עם כשלונות', דיברתי על...", "אני ממליצה...", "אני חושבת ש...".
  Never use the plural "we" for yourself ("אנחנו ממליצים", "דיברנו"), and do not say "the podcast
  says".
- Use only what the passages say. If they do not address the question, say plainly that you have
  not talked about it on the show, and return "cited": []. Do not stretch weak matches.
- Develop the answer: open by relating to the listener's specific situation, then give 2-4
  practical points (the idea behind each, how it applies here, what they could try), then one
  closing sentence. All of it only as far as the passages support it.
- Write in your own words. Do not copy sentences from the passages; at most quote a few words.
- Synthesize across episodes rather than listing passages. When you bring an idea from an episode,
  name the episode in the text (number and title) and put a marker like [3] after the claim, with
  the number of the passage it comes from.
- There are no speaker labels: never name a guest or a co-host, and never claim a specific person
  said something. Present the ideas as what came up in the episode, in your own voice.
- Transcripts are automatic, so names may be misspelled. Do not correct them with guesses.
- If you add general knowledge, mark it clearly as not from the show.
- Formatting: paragraphs separated by a blank line, lines starting with "- " for bullets, **bold**.
  Nothing else (no headings, no tables, no links).
- Aim for about 300-450 words."""

DEFAULTS = {
    "queries": 3,
    "top_per_query": 20,
    "context_chunks": 15,
    "neighbours": True,
    "max_context_chars": 36000,
    "fallback_episodes": 5,
    "expand_thinking_budget": None,
    "answer_max_tokens": 4096,
}

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Reference:
    number: int            # the marker number used in the answer text ([1], [2], ...)
    episode_id: int
    title: str
    pub_date: str
    t_start: int
    spotify_url: str = ""

    @property
    def time(self) -> str:
        return format_time(self.t_start)


@dataclass
class AskResult:
    question: str                       # as the user asked it
    standalone_question: str            # rewritten to stand on its own (same as question if no history)
    queries: list[str]
    answer: Optional[str]               # None when the LLM was unavailable (fallback)
    references: list[Reference] = field(default_factory=list)
    fallback: bool = False
    error: str = ""
    coverage: str = ""                  # "149 of 252 episodes are searchable"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for ref, raw in zip(self.references, data["references"]):
            raw["time"] = ref.time
        return data


@dataclass
class Turn:
    """A previous exchange, for follow-ups."""
    question: str
    answer: str

    @classmethod
    def after(cls, result: "AskResult") -> "Turn":
        """The exchange a follow-up builds on. Without a written answer the follow-up still gets
        the question, so it keeps the topic."""
        return cls(result.standalone_question, result.answer or "(no answer was given)")


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    cfg = section(config, "ask")
    return {key: cfg.get(key, default) for key, default in DEFAULTS.items()}


# ---------------------------------------------------------------- steps


def expand_question(llm: LLM, question: str, history: Optional[Turn], n: int,
                    thinking_budget: Optional[int] = None) -> tuple[str, list[str]]:
    """Step 1: standalone question + n Hebrew search queries."""
    prompt = ""
    if history:
        prompt += f"Previous question: {history.question}\nPrevious answer: {history.answer}\n\n"
    prompt += f"Question: {question}"
    data = parse_json(llm.generate(EXPAND_SYSTEM.format(n=n), prompt, json_output=True,
                                   max_output_tokens=1024, thinking_budget=thinking_budget))
    if not isinstance(data, dict):
        raise LLMError("Query expansion did not return a JSON object")
    standalone = str(data.get("question") or question).strip() or question
    queries = [str(q).strip() for q in data.get("queries") or [] if str(q).strip()][:n]
    return standalone, queries


def merge_hits(hit_lists: Sequence[Sequence[ChunkHit]]) -> list[ChunkHit]:
    """Union of several searches: one entry per chunk with its best score, best first."""
    best: dict[int, ChunkHit] = {}
    for hits in hit_lists:
        for hit in hits:
            if hit.chunk_id not in best or hit.score > best[hit.chunk_id].score:
                best[hit.chunk_id] = hit
    return sorted(best.values(), key=lambda h: -h.score)


def build_context(index: VectorIndex, hits: Sequence[ChunkHit], episodes: dict[int, csvstore.Episode],
                  limit: int, neighbours: bool, max_chars: int) -> tuple[str, list[ChunkHit]]:
    """Numbered passages for the LLM. Each is a chunk plus (optionally) the chunk after it, so a
    question and its answer that straddle a chunk boundary stay together."""
    by_position = {(row[1], row[2]): row for row in index.rows}   # (episode_id, chunk_index) -> row
    used: list[ChunkHit] = []
    parts: list[str] = []
    total = 0
    for hit in hits[:limit]:
        text = hit.text
        if neighbours:
            following = by_position.get((hit.episode_id, hit.chunk_index + 1))
            if following:
                text = f"{text}\n{following[5]}"
        episode = episodes.get(hit.episode_id)
        title = episode.title if episode else f"episode {hit.episode_id}"
        date = episode.pub_date if episode else ""
        block = f"[{len(used) + 1}] Episode #{hit.episode_id} \"{title}\" ({date}) at {format_time(hit.t_start)}\n{text}"
        if used and total + len(block) > max_chars:
            break
        parts.append(block)
        used.append(hit)
        total += len(block)
    return "\n\n".join(parts), used


def _reference(number: int, hit: ChunkHit, episodes: dict[int, csvstore.Episode]) -> Reference:
    episode = episodes.get(hit.episode_id)
    return Reference(
        number=number,
        episode_id=hit.episode_id,
        title=episode.title if episode else f"episode {hit.episode_id}",
        pub_date=episode.pub_date if episode else "",
        t_start=hit.t_start,
        spotify_url=episode.spotify_url if episode else "",
    )


def renumber_citations(answer: str, cited: Sequence[Any], n_passages: int) -> tuple[str, list[int]]:
    """Keep only markers that point at real passages, renumber them 1..k in order of first use,
    and return the passage numbers (1-based, original numbering) in that order."""
    valid = set()
    for value in cited:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= n_passages:
            valid.add(number)
    order: list[int] = []

    def replace(match: re.Match) -> str:
        new = []
        for part in match.group(1).split(","):
            number = int(part)
            if 1 <= number <= n_passages:
                valid.add(number)   # a marker in the text counts as cited even if "cited" forgot it
                if number not in order:
                    order.append(number)
                new.append(str(order.index(number) + 1))
        return f"[{', '.join(new)}]" if new else ""

    text = _MARKER.sub(replace, answer)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)   # space left before punctuation by a dropped marker
    order += sorted(valid - set(order))   # cited but never marked: still listed, at the end
    return text.strip(), order


def _fallback_references(hits: Sequence[ChunkHit], episodes: dict[int, csvstore.Episode], limit: int) -> list[Reference]:
    """Best hit of each of the top episodes, for when there is no written answer."""
    refs: list[Reference] = []
    seen: set[int] = set()
    for hit in hits:
        if hit.episode_id in seen:
            continue
        seen.add(hit.episode_id)
        refs.append(_reference(len(refs) + 1, hit, episodes))
        if len(refs) >= limit:
            break
    return refs


def coverage_line(index: VectorIndex, episodes: dict[int, csvstore.Episode]) -> str:
    searchable = len(set(int(e) for e in np.unique(index.episode_ids)))
    total = sum(1 for e in episodes.values() if e.title)
    return f"{searchable} of {total} episodes are searchable"


# ---------------------------------------------------------------- pipeline


class Asker:
    """Holds the index, embedder and LLM for one show, so several questions reuse them."""

    def __init__(self, config: dict[str, Any], paths: ShowPaths,
                 embedder: Optional[Embedder] = None, llm: Optional[LLM] = None):
        self.settings = _settings(config)
        self.embedder = embedder or build_embedder(config)
        self.llm = llm or build_llm(config)
        self.index = VectorIndex(paths.index_file(self.embedder.model, self.embedder.dimensions))
        self.episodes = {e.id: e for e in csvstore.read_episodes(paths.episodes_csv)}
        self.coverage = coverage_line(self.index, self.episodes)

    def ask(self, question: str, history: Optional[Turn] = None) -> AskResult:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        s = self.settings

        # 1. expand (on failure, search with the question alone and still try to answer)
        try:
            standalone, queries = expand_question(self.llm, question, history, int(s["queries"]),
                                                  s["expand_thinking_budget"])
        except LLMError:
            standalone, queries = question, []

        # 2. embed (one request) + 3. search and merge
        texts = list(dict.fromkeys([standalone, *queries]))
        vectors = self.embedder.embed(texts)
        hits = merge_hits([self.index.search(v, int(s["top_per_query"])) for v in vectors])

        result = AskResult(question, standalone, queries, None, coverage=self.coverage)

        # 4. answer
        context, used = build_context(self.index, hits, self.episodes, int(s["context_chunks"]),
                                      bool(s["neighbours"]), int(s["max_context_chars"]))
        prompt = ""
        if history:
            prompt += f"Previous question: {history.question}\nPrevious answer: {history.answer}\n\n"
        prompt += f"Passages:\n\n{context}\n\nQuestion: {standalone}"
        try:
            data = parse_json(self.llm.generate(ANSWER_SYSTEM, prompt, json_output=True,
                                                max_output_tokens=int(s["answer_max_tokens"])))
            if not isinstance(data, dict) or not str(data.get("answer") or "").strip():
                raise LLMError("Answer step returned no answer")
        except LLMError as exc:
            return self._fallback(result, hits, str(exc))

        # 5. references
        answer, order = renumber_citations(str(data["answer"]), data.get("cited") or [], len(used))
        result.answer = answer
        result.references = [_reference(n, used[p - 1], self.episodes) for n, p in enumerate(order, start=1)]
        return result

    def _fallback(self, result: AskResult, hits: Sequence[ChunkHit], error: str) -> AskResult:
        result.fallback = True
        result.error = error
        result.references = _fallback_references(hits, self.episodes, int(self.settings["fallback_episodes"]))
        return result


# ---------------------------------------------------------------- output


def format_result(result: AskResult) -> str:
    lines = []
    if result.standalone_question != result.question:
        lines.append(f"(understood as: {result.standalone_question})")
    if result.fallback:
        lines.append(f"No written answer: {result.error}")
        lines.append("Episodes that seem most relevant:")
    else:
        lines.append(result.answer or "")
        if result.references:
            lines.append("\nReferences:")
    for ref in result.references:
        marker = "-" if result.fallback else f"[{ref.number}]"
        link = f"  {ref.spotify_url}" if ref.spotify_url else ""
        lines.append(f"{marker} #{ref.episode_id} {ref.title} ({ref.pub_date}) [{ref.time}]{link}")
    lines.append(f"\n{result.coverage}. Answers are written automatically from automatic transcripts.")
    return "\n".join(lines)


def render_page(results: Sequence[AskResult], show_title: str, model: str) -> str:
    """One HTML page for one or more answers (a question and its follow-up). No transcript text."""
    from podcast_skill.report import STYLE, render_answer

    sections = []
    for result in results:
        if result.fallback:
            body = ('<div class="answer"><div class="label">אין תשובה כתובה כרגע. אלה הפרקים שנראים הכי רלוונטיים:'
                    "</div></div>")
        else:
            body = render_answer(result.answer or "")
        refs = []
        for ref in result.references:
            marker = "" if result.fallback else f"[{ref.number}] "
            title = html.escape(f"#{ref.episode_id} {ref.title}")
            if ref.spotify_url:
                title = f'<a href="{html.escape(ref.spotify_url)}">{title}</a>'
            refs.append(f"<li>{marker}{title} · {html.escape(ref.pub_date)} · {ref.time}</li>")
        sections.append(
            f"<section><h2>{html.escape(result.question)}</h2>{body}"
            + (f'<ul class="refs">{"".join(refs)}</ul>' if refs else "")
            + "</section>"
        )
    coverage = results[0].coverage if results else ""
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        '<!doctype html>\n<html lang="he" dir="rtl"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(show_title)}</title>"
        f"<style>{STYLE} ul.refs {{ padding-inline-start:1.3em; }} ul.refs li {{ margin:.3em 0; }}</style></head><body>"
        f"<h1>{html.escape(show_title)}</h1>"
        f'<div class="meta">{html.escape(coverage)} · התשובות נכתבו אוטומטית ({html.escape(model)}) '
        f"מתמלולים אוטומטיים · {stamp}</div>"
        + "\n".join(sections)
        + "</body></html>\n"
    )


def write_page(paths: ShowPaths, results: Sequence[AskResult], model: str, out: Optional[Path] = None) -> Path:
    show_title = paths.slug
    if paths.show_json.exists():
        import json

        show_title = json.loads(paths.show_json.read_text(encoding="utf-8")).get("title") or show_title
    out = out or paths.root / "answers" / f"answer-{dt.datetime.now():%Y%m%d-%H%M%S}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(results, show_title, model), encoding="utf-8")
    return out
