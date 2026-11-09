"""Search report: run a list of questions and write the best passages for each to an HTML page.

Meant for judging retrieval quality by eye: one section per question, a table of the top
passages with their cosine-similarity score, episode, and time in the episode.
"""
from __future__ import annotations

import datetime as dt
import html
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from podcast_skill import episodes_csv as csvstore
from podcast_skill.chunking import format_time
from podcast_skill.embeddings import Embedder
from podcast_skill.index import ChunkHit, search_chunks
from podcast_skill.paths import ShowPaths

STYLE = """
:root { --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#ddd; --card:#f7f7f8; --bar:#3b6fd4; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16171a; --fg:#e8e8ea; --muted:#9a9aa2; --line:#33353b; --card:#1f2024; --bar:#6c97f0; }
}
body { background:var(--bg); color:var(--fg); font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
       max-width:1000px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:1.4rem; margin:0 0 4px; }
.meta { color:var(--muted); font-size:.9rem; margin-bottom:28px; }
section { margin:0 0 36px; }
h2 { font-size:1.15rem; margin:0 0 10px; padding:10px 14px; background:var(--card); border-radius:8px; }
h2 span { color:var(--muted); font-weight:normal; margin-inline-end:8px; }
table { width:100%; border-collapse:collapse; }
th, td { text-align:start; vertical-align:top; padding:8px 10px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-size:.8rem; font-weight:600; }
td.rank { width:2em; color:var(--muted); }
td.score { width:5.5em; white-space:nowrap; font-variant-numeric:tabular-nums; }
.bar { height:4px; background:var(--bar); border-radius:2px; margin-top:4px; }
td.where { width:13em; font-size:.88rem; }
td.where .ep { font-weight:600; }
td.where .sub { color:var(--muted); }
td.text { white-space:pre-wrap; }
.answer { margin:0 0 14px; padding:12px 16px; border-inline-start:4px solid var(--bar); background:var(--card); border-radius:6px; }
.answer .label { color:var(--muted); font-size:.8rem; font-weight:600; margin-bottom:6px; }
.answer p { margin:.4em 0; }
.answer ul { margin:.4em 0; padding-inline-start:1.3em; }
.answer li { margin:.35em 0; }
@media (max-width:640px) { td.where { width:8em; } body { font-size:15px; } }
"""


def _row(rank: int, hit: ChunkHit, episodes: dict[int, csvstore.Episode], bar_scale: float) -> str:
    episode = episodes.get(hit.episode_id)
    title = episode.title if episode else f"episode {hit.episode_id}"
    date = episode.pub_date if episode else ""
    width = max(0.0, min(100.0, hit.score / bar_scale * 100))
    return (
        "<tr>"
        f'<td class="rank">{rank}</td>'
        f'<td class="score">{hit.score:.3f}<div class="bar" style="width:{width:.0f}%"></div></td>'
        f'<td class="where"><div class="ep">#{hit.episode_id} {html.escape(title)}</div>'
        f'<div class="sub">{html.escape(date)} · {format_time(hit.t_start)}</div></td>'
        f'<td class="text">{html.escape(hit.text)}</td>'
        "</tr>"
    )


def parse_answers(text: str) -> list[str]:
    """Answers in one file, one per question, separated by a line containing only `===`."""
    return [part.strip() for part in re.split(r"^===\s*$", text, flags=re.MULTILINE)]


def _inline(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))


def render_answer(text: str) -> str:
    """Tiny markup: blank-line separated paragraphs, `- ` bullets, **bold**. Everything is escaped."""
    if not text.strip():
        return ""
    blocks, bullets = [], []

    def flush() -> None:
        if bullets:
            blocks.append("<ul>" + "".join(f"<li>{_inline(b)}</li>" for b in bullets) + "</ul>")
            bullets.clear()

    for line in text.splitlines():
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        elif line.strip():
            flush()
            blocks.append(f"<p>{_inline(line.strip())}</p>")
    flush()
    return ('<div class="answer"><div class="label">תשובה — נכתבה על סמך הקטעים והטקסט שסביבם, '
            "לא ציטוט מהפודקאסט</div>" + "".join(blocks) + "</div>")


def render_report(
    questions: Sequence[str],
    results: Sequence[Sequence[ChunkHit]],
    episodes: dict[int, csvstore.Episode],
    show_title: str,
    model: str,
    answers: Optional[Sequence[str]] = None,
) -> str:
    best = max((hit.score for hits in results for hit in hits), default=1.0) or 1.0
    sections = []
    for number, (question, hits) in enumerate(zip(questions, results), start=1):
        rows = "\n".join(_row(rank, hit, episodes, best) for rank, hit in enumerate(hits, start=1))
        answer = render_answer(answers[number - 1]) if answers and number <= len(answers) else ""
        sections.append(
            f"<section><h2><span>{number}.</span>{html.escape(question)}</h2>{answer}"
            "<table><thead><tr><th>#</th><th>Score</th><th>Episode · time</th><th>Passage</th></tr></thead>"
            f"<tbody>\n{rows}\n</tbody></table></section>"
        )
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        '<!doctype html>\n<html lang="he" dir="rtl"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Search report</title>"
        f"<style>{STYLE}</style></head><body>"
        f"<h1>Search report</h1>"
        f'<div class="meta">{html.escape(show_title)} · model {html.escape(model)} · {stamp} · '
        "score = cosine similarity of question and passage (higher is closer; not a percentage)</div>"
        + "\n".join(sections)
        + "</body></html>\n"
    )


def write_report(
    config: dict[str, Any],
    paths: ShowPaths,
    questions: Sequence[str],
    top: int = 5,
    out: Optional[Path] = None,
    embedder: Optional[Embedder] = None,
    answers: Optional[Sequence[str]] = None,
) -> Path:
    from podcast_skill.embeddings import build_embedder

    embedder = embedder or build_embedder(config)
    results = search_chunks(config, paths, questions, top, embedder)
    episodes = {e.id: e for e in csvstore.read_episodes(paths.episodes_csv)}
    show_title = paths.slug
    if paths.show_json.exists():
        import json

        show_title = json.loads(paths.show_json.read_text(encoding="utf-8")).get("title") or show_title
    page = render_report(questions, results, episodes, show_title, embedder.model, answers)
    out = out or paths.reports_dir / f"search-report-{dt.datetime.now():%Y%m%d-%H%M%S}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
