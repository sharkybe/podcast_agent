from __future__ import annotations

from typing import Protocol

from podcast_skill.episodes_csv import Episode


class TranscriptProvider(Protocol):
    """A source of transcripts. Providers are tried in order by step 2.

    `fetch` returns the cleaned transcript lines, or None when this provider has
    no transcript for the episode. It may raise on unexpected errors.
    """

    name: str

    def open(self) -> None: ...

    def close(self) -> None: ...

    def fetch(self, episode: Episode) -> list[str] | None: ...
