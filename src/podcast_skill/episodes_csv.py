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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for episode in sorted(episodes, key=lambda e: e.id):
            writer.writerow(episode.to_row())


def read_episodes(path: Path) -> list[Episode]:
    episodes: list[Episode] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            data = {name: (row.get(name) or "") for name in FIELDNAMES}
            data["id"] = int(data["id"])
            data["status"] = data["status"] or PENDING
            episodes.append(Episode(**data))
    return sorted(episodes, key=lambda e: e.id)


def merge_with_existing(new: list[Episode], existing: list[Episode]) -> list[Episode]:
    """Keep progress (status, transcript, spotify link) of episodes already in the CSV.

    Episodes are matched by audio_url, falling back to title. New episodes are
    appended with fresh ids; existing ids never change.
    """
    by_audio = {e.audio_url: e for e in existing if e.audio_url}
    by_title = {e.title: e for e in existing}
    merged: dict[int, Episode] = {e.id: e for e in existing}
    next_id = max(merged, default=0) + 1

    for episode in new:  # `new` is ordered oldest -> newest
        match = by_audio.get(episode.audio_url) or by_title.get(episode.title)
        if match is not None:
            for name in ("episode_number", "pub_date", "duration", "audio_url", "rss_transcript_url"):
                value = getattr(episode, name)
                if value:
                    setattr(match, name, value)
            continue
        episode.id = next_id
        next_id += 1
        merged[episode.id] = episode
    return sorted(merged.values(), key=lambda e: e.id)
