"""Step 2: fetch transcripts for every episode in episodes.csv.

Each episode ends up `done` (transcript saved) or `unavailable` (no provider had a
transcript). Unexpected errors mark it `failed`, which is retried on the next run.
"""
from __future__ import annotations

from typing import Any, Sequence

from podcast_skill import episodes_csv as csvstore
from podcast_skill.config import section
from podcast_skill.episodes_csv import Episode
from podcast_skill.paths import ShowPaths
from podcast_skill.providers import TranscriptProvider


def build_providers(
    config: dict[str, Any],
    paths: ShowPaths | None = None,
    names: Sequence[str] | None = None,
) -> list[TranscriptProvider]:
    names = names or section(config, "transcribe").get("providers") or ["spotify"]
    providers: list[TranscriptProvider] = []
    for name in names:
        if name == "spotify":
            from podcast_skill.providers.spotify import SpotifyTranscriptProvider

            providers.append(SpotifyTranscriptProvider(config, debug_dir=paths.debug_dir if paths else None))
        elif name == "ivrit":
            from podcast_skill.providers.ivrit import IvritTranscriptProvider

            providers.append(IvritTranscriptProvider(config, audio_dir=paths.audio_dir if paths else None))
        else:
            raise ValueError(f"Unknown transcript provider: {name}")
    return providers


def select_episodes(
    episodes: list[Episode],
    retry_unavailable: bool = False,
    limit: int | None = None,
    only_ids: Sequence[int] | None = None,
) -> list[Episode]:
    if only_ids:  # explicit request: run these whatever their current status
        wanted_ids = set(only_ids)
        return [e for e in episodes if e.id in wanted_ids]
    wanted = {csvstore.PENDING, csvstore.FAILED}
    if retry_unavailable:
        wanted.add(csvstore.UNAVAILABLE)
    todo = [e for e in episodes if e.status in wanted]
    return todo[:limit] if limit else todo


def transcribe(
    config: dict[str, Any],
    paths: ShowPaths,
    providers: Sequence[TranscriptProvider] | None = None,
    retry_unavailable: bool | None = None,
    limit: int | None = None,
    only_ids: Sequence[int] | None = None,
    provider_names: Sequence[str] | None = None,
) -> dict[str, int]:
    cfg = section(config, "transcribe")
    if retry_unavailable is None:
        retry_unavailable = bool(cfg.get("retry_unavailable", False))
    if limit is None:
        limit = cfg.get("limit")
    if not paths.episodes_csv.exists():
        raise FileNotFoundError(f"{paths.episodes_csv} not found: run the discover step first")

    episodes = csvstore.read_episodes(paths.episodes_csv)
    todo = select_episodes(episodes, retry_unavailable, limit, only_ids)
    counts = {csvstore.DONE: 0, csvstore.UNAVAILABLE: 0, csvstore.FAILED: 0}
    if not todo:
        print("Nothing to do: every episode is done or unavailable.")
        return counts

    providers = list(providers) if providers is not None else build_providers(config, paths, provider_names)
    paths.transcripts_dir.mkdir(parents=True, exist_ok=True)

    opened: list[TranscriptProvider] = []
    try:
        for provider in providers:
            provider.open()
            opened.append(provider)

        for position, episode in enumerate(todo, start=1):
            print(f"[{position}/{len(todo)}] #{episode.id} {episode.title}")
            lines, source, error = None, "", None
            for provider in providers:
                try:
                    lines = provider.fetch(episode)
                except Exception as exc:  # keep going: one bad episode must not stop the run
                    error = exc
                    print(f"  WARNING: {provider.name} failed: {exc}")
                    continue
                if lines:
                    source = provider.name
                    break

            if lines:
                out_file = paths.transcript_file(episode.id)
                out_file.write_text("\n".join(lines), encoding="utf-8")
                episode.transcript_path = str(out_file.relative_to(paths.root))
                episode.status = csvstore.DONE
                print(f"  saved {len(lines)} lines via {source}")
            elif error is not None:
                episode.status = csvstore.FAILED
            else:
                episode.transcript_path = ""
                episode.status = csvstore.UNAVAILABLE
                print("  no transcript available")
            counts[episode.status] += 1
            csvstore.write_episodes(paths.episodes_csv, episodes)  # checkpoint after each episode
    finally:
        for provider in opened:
            try:
                provider.close()
            except Exception:
                pass

    print(f"Done: {counts[csvstore.DONE]} transcribed, {counts[csvstore.UNAVAILABLE]} unavailable, "
          f"{counts[csvstore.FAILED]} failed.")
    return counts
