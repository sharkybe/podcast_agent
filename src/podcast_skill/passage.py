"""Read a stretch of one transcript (to get more context around a search hit)."""
from __future__ import annotations

from typing import Optional

from podcast_skill import episodes_csv as csvstore
from podcast_skill.chunking import format_time, parse_turns
from podcast_skill.paths import ShowPaths

DEFAULT_SPAN_SECONDS = 180


def parse_time(value: str) -> int:
    """'90', '1:30' (m:ss) or '1:02:03' (h:mm:ss) -> seconds."""
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Bad time {value!r}: use seconds, m:ss or h:mm:ss")
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds


def get_passage(
    paths: ShowPaths,
    episode_id: int,
    t_from: int,
    t_to: Optional[int] = None,
) -> str:
    """Speaker turns of the episode that start in [t_from, t_to], one `[m:ss] text` block per turn.

    The turn that started just before `t_from` is included too, because it may still be
    running at `t_from`.
    """
    episodes = {e.id: e for e in csvstore.read_episodes(paths.episodes_csv)}
    episode = episodes.get(episode_id)
    if episode is None:
        raise ValueError(f"No episode with id {episode_id}")
    if not episode.transcript_path or not (paths.root / episode.transcript_path).exists():
        raise FileNotFoundError(f"Episode {episode_id} has no transcript (status: {episode.status})")
    if t_to is None:
        t_to = t_from + DEFAULT_SPAN_SECONDS
    if t_to < t_from:
        raise ValueError("The end time is before the start time")

    turns = parse_turns((paths.root / episode.transcript_path).read_text(encoding="utf-8"))
    before = [turn for turn in turns if turn[0] < t_from][-1:]
    inside = [turn for turn in turns if t_from <= turn[0] <= t_to]
    return "\n".join(f"[{format_time(t)}] " + "\n".join(lines) for t, lines in before + inside)
