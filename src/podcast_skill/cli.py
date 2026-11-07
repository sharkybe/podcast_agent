from __future__ import annotations

import argparse
import sys

from podcast_skill import discover as discover_step
from podcast_skill import transcribe as transcribe_step
from podcast_skill.config import load_config
from podcast_skill.paths import ShowPaths, artifacts_dir


def _show_paths(config, show: str | None) -> ShowPaths:
    shows = sorted(csv_file.parent.name for csv_file in artifacts_dir(config).glob("*/episodes.csv"))
    if show:
        return ShowPaths(config, show)
    if len(shows) == 1:
        return ShowPaths(config, shows[0])
    if not shows:
        raise SystemExit("No shows found in artifacts/. Run `discover` first.")
    raise SystemExit("Several shows found, pass --show <slug>: " + ", ".join(shows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="podcast_skill", description="Podcast skill: steps 1-2")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("discover", help="Step 1: find the show and list its episodes")
    d.add_argument("query", help="Podcast name or a Spotify show link")
    d.add_argument("--pick", type=int, help="Choose candidate N non-interactively (1-based)")
    d.add_argument("--spotify-pick", type=int, help="Choose the Spotify show candidate N (1-based)")
    d.add_argument("--feed-url", help="Skip the search and use this RSS feed")
    d.add_argument("--no-spotify", action="store_true", help="Do not link episodes to Spotify")

    t = sub.add_parser("transcribe", help="Step 2: fetch transcripts for the show's episodes")
    t.add_argument("--show", help="Show slug under artifacts/ (optional if only one)")
    t.add_argument("--limit", type=int, help="Max episodes this run")
    t.add_argument("--episode", type=int, action="append", help="Only this episode id (repeatable); ignores status")
    t.add_argument("--retry-unavailable", action="store_true", help="Retry episodes marked unavailable")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()

    if args.command == "discover":
        try:
            discover_step.discover(
                config,
                args.query,
                pick=args.pick,
                feed_url=args.feed_url,
                spotify_pick=args.spotify_pick,
                interactive=sys.stdin.isatty(),
                link_to_spotify=False if args.no_spotify else None,
            )
        except discover_step.NeedsChoice as need:
            print(f"\nCONFIRMATION NEEDED: found several {need.label} candidates:")
            for number, option in enumerate(need.candidates, start=1):
                print(f"  {number}. {option}")
            flag = "--spotify-pick" if "Spotify" in need.label else "--pick"
            print(f"\nAsk the user which is correct, then re-run with {flag} N.")
            return 2
        return 0

    if args.command == "transcribe":
        paths = _show_paths(config, args.show)
        transcribe_step.transcribe(
            config,
            paths,
            retry_unavailable=True if args.retry_unavailable else None,
            limit=args.limit,
            only_ids=args.episode,
        )
        return 0
    return 1
