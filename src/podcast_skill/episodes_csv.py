from __future__ import annotations

import csv
from dataclasses import dataclass, field, fields
from pathlib import Path

PENDING = "pending"
DONE = "done"
UNAVAILABLE = "unavailable"
FAILED = "failed"
STATUSES = (PENDING, DONE, UNAVAILABLE, FAILED)


@dataclass
class Episode:
    id: int                      # 1 = oldest; stable key, also the transcript file name
    title: str
    episode_number: str = ""     # number published by the show (may be empty)
    pub_date: str = ""           # ISO date
    duration: str = ""
    audio_url: str = ""
    rss_transcript_url: str = ""
    spotify_url: str = ""
    status: str = PENDING
    transcript_path: str = ""    # relative to the show artifacts dir

    def to_row(self) -> dict[str, str]:
        return {f.name: str(getattr(self, f.name)) for f in fields(self)}


FIELDNAMES = [f.name for f in fields(Episode)]


def write_episodes(path: Path, episodes: list[Episode]) -> None:
    """Write one row per id from 1 to the highest id, so row N (after the header) is episode N.

    Ids with no episode (a number the show skipped) get an empty row holding only the id.
    """
    by_id: dict[int, Episode] = {}
    for episode in episodes:
        if episode.id in by_id:
            raise ValueError(f"Two episodes share id {episode.id}: {by_id[episode.id].title!r} and {episode.title!r}")
        by_id[episode.id] = episode
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for episode_id in range(1, max(by_id, default=0) + 1):
            if episode_id in by_id:
                writer.writerow(by_id[episode_id].to_row())
            else:
                writer.writerow({**{name: "" for name in FIELDNAMES}, "id": str(episode_id)})


def _number(episode: Episode) -> int | None:
    text = (episode.episode_number or "").strip()
    return int(text) if text.isdigit() and int(text) > 0 else None


def assign_ids(episodes: list[Episode]) -> None:
    """Set every episode's id so that the id IS the episode number from its title.

    - Numbered episodes get id = number (the first one wins if a number appears twice).
    - An unnumbered episode takes a missing number only when it is unambiguous: exactly one
      number is missing between its two dated neighbours and it is the only unnumbered
      episode there (e.g. an untitled intro published before episode 2 becomes episode 1).
    - Everything else (bonus episodes) goes after the highest number, oldest first.
    - Numbers nobody has stay free: the CSV writer leaves an empty row for them.
    """
    by_date = sorted(episodes, key=lambda e: (not e.pub_date, e.pub_date, e.id))  # undated ones last
    numbered: dict[int, Episode] = {}
    loose: list[Episode] = []
    for episode in by_date:
        number = _number(episode)
        if number is None or number in numbered:
            loose.append(episode)
        else:
            numbered[number] = episode

    anchors = [(episode.pub_date, number) for number, episode in numbered.items()]
    windows: dict[tuple[int, int], list[Episode]] = {}
    for episode in loose:
        if not episode.pub_date:
            continue  # no date, no way to tell which gap it belongs to
        before =[n for date, n in anchors if date <= episode.pub_date]
        after = [n for date, n in anchors if date > episode.pub_date]
        if after:
            windows.setdefault((max(before, default=0), min(after)), []).append(episode)

    assigned = dict(numbered)
    for (low, high), group in windows.items():
        free = [n for n in range(low + 1, high) if n not in assigned]
        if len(group) == 1 and len(free) == 1:
            assigned[free[0]] = group[0]

    placed = {id(episode) for episode in assigned.values()}
    for number, episode in assigned.items():
        episode.id = number
    next_id = max(assigned, default=0) + 1
    for episode in loose:
        if id(episode) not in placed:
            episode.id = next_id
            next_id += 1


def read_episodes(path: Path) -> list[Episode]:
    """Read the real episodes; the empty placeholder rows for skipped numbers are left out."""
    episodes: list[Episode] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not (row.get("title") or "").strip():
                continue
            data = {name: (row.get(name) or "") for name in FIELDNAMES}
            data["id"] = int(data["id"])
            data["status"] = data["status"] or PENDING
            episodes.append(Episode(**data))
    return sorted(episodes, key=lambda e: e.id)


def merge_with_existing(new: list[Episode], existing: list[Episode]) -> list[Episode]:
    """Keep progress (status, transcript, spotify link) of episodes already in the CSV.

    Episodes are matched by audio_url, falling back to title. Ids are then recomputed from
    the episode numbers (`assign_ids`). An episode whose id changes loses its status and
    transcript path (back to pending): the transcript file under the old id is not its own.
    """
    by_audio = {e.audio_url: e for e in existing if e.audio_url}
    by_title = {e.title: e for e in existing}
    merged: list[Episode] = list(existing)
    old_ids = {id(e): e.id for e in existing}

    for episode in new:
        match = by_audio.get(episode.audio_url) or by_title.get(episode.title)
        if match is not None:
            for name in ("episode_number", "pub_date", "duration", "audio_url", "rss_transcript_url"):
                value = getattr(episode, name)
                if value:
                    setattr(match, name, value)
            continue
        merged.append(episode)

    assign_ids(merged)
    for episode in existing:
        if episode.id != old_ids[id(episode)]:
            episode.status = PENDING
            episode.transcript_path = ""
    return sorted(merged, key=lambda e: e.id)
