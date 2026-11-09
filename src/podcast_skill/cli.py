from __future__ import annotations

import argparse
import json
import sys

from podcast_skill import discover as discover_step
from podcast_skill import index as index_step
from podcast_skill import transcribe as transcribe_step
from podcast_skill.config import load_config, load_dotenv, section
from podcast_skill.embeddings import build_embedder
from pathlib import Path

from podcast_skill.passage import get_passage, parse_time
from podcast_skill.report import parse_answers, write_report
from podcast_skill.terminal import fix_rtl_output
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
    parser = argparse.ArgumentParser(prog="podcast_skill", description="Podcast skill: discover, transcribe, index, search")
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
    t.add_argument("--providers", help="Comma-separated providers, overrides transcribe.providers (e.g. ivrit, or spotify,ivrit)")

    i = sub.add_parser("index", help="Step 3: embed the transcripts (incremental)")
    i.add_argument("--show", help="Show slug under artifacts/ (optional if only one)")
    i.add_argument("--episode", type=int, action="append", help="Only this episode id (repeatable); re-embeds it")
    i.add_argument("--rebuild", action="store_true", help="Delete this model's index and embed everything again")
    i.add_argument("--model", help="Embedding model, overrides index.embedding_model (e.g. text-embedding-3-small)")

    s = sub.add_parser("search", help="Step 3: semantic search over the indexed transcripts")
    s.add_argument("query", help="What to look for, in natural language")
    s.add_argument("--show", help="Show slug under artifacts/ (optional if only one)")
    s.add_argument("-k", "--top-chunks", type=int, help="How many passages to retrieve (default: index.top_chunks)")
    s.add_argument("--episode", type=int, action="append", help="Only search this episode id (repeatable)")
    s.add_argument("--min-score", type=float, help="Drop passages scoring below this")
    s.add_argument("--model", help="Embedding model, overrides index.embedding_model")
    s.add_argument("--json", action="store_true", help="Print full results as JSON")

    r = sub.add_parser("report", help="Run questions and write the best passages for each to an HTML page")
    r.add_argument("questions", nargs="*", help="Questions (or use --file)")
    r.add_argument("--file", help="Text file with one question per line")
    r.add_argument("--answers", help="Text file with one answer per question, separated by lines containing only ===")
    r.add_argument("--top", type=int, default=5, help="Passages per question (default 5)")
    r.add_argument("--out", help="Output .html path (default: artifacts/<show>/reports/)")
    r.add_argument("--show", help="Show slug under artifacts/ (optional if only one)")
    r.add_argument("--model", help="Embedding model, overrides index.embedding_model")

    p = sub.add_parser("passage", help="Read part of one transcript (more context around a search hit)")
    p.add_argument("episode", type=int, help="Episode id")
    p.add_argument("--from", dest="t_from", required=True, help="Start time: seconds, m:ss or h:mm:ss")
    p.add_argument("--to", dest="t_to", help="End time (default: start + 3 minutes)")
    p.add_argument("--show", help="Show slug under artifacts/ (optional if only one)")

    st = sub.add_parser("status", help="How many episodes are transcribed and indexed")
    st.add_argument("--show", help="Show slug under artifacts/ (optional if only one)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()  # API keys from .env (gitignored); real environment variables take precedence
    config = load_config()
    if not getattr(args, "json", False):  # JSON must stay in logical order
        fix_rtl_output(bool(section(config, "display").get("fix_rtl", True)))

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
            provider_names=[p.strip() for p in args.providers.split(",") if p.strip()] if args.providers else None,
        )
        return 0

    if args.command == "index":
        paths = _show_paths(config, args.show)
        try:
            index_step.index_show(
                config,
                paths,
                embedder=build_embedder(config, args.model),
                only_ids=args.episode,
                rebuild=args.rebuild,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            raise SystemExit(f"ERROR: {exc}")
        return 0

    if args.command == "search":
        paths = _show_paths(config, args.show)
        try:
            results = index_step.search_episodes(
                config,
                paths,
                args.query,
                top_chunks=args.top_chunks,
                episode_ids=args.episode,
                min_score=args.min_score,
                embedder=build_embedder(config, args.model),
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"ERROR: {exc}")
        if args.json:
            print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        else:
            print(index_step.format_hits(results))
        return 0

    if args.command == "report":
        paths = _show_paths(config, args.show)
        questions = list(args.questions)
        if args.file:
            questions += [line.strip() for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not questions:
            raise SystemExit("ERROR: give questions as arguments or with --file")
        try:
            out = write_report(
                config, paths, questions, top=args.top,
                out=Path(args.out) if args.out else None,
                embedder=build_embedder(config, args.model),
                answers=parse_answers(Path(args.answers).read_text(encoding="utf-8")) if args.answers else None,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"ERROR: {exc}")
        print(f"Report written: {out}")
        return 0

    if args.command == "passage":
        paths = _show_paths(config, args.show)
        try:
            print(get_passage(
                paths,
                args.episode,
                parse_time(args.t_from),
                parse_time(args.t_to) if args.t_to else None,
            ))
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"ERROR: {exc}")
        return 0

    if args.command == "status":
        print(index_step.status_report(_show_paths(config, args.show)))
        return 0
    return 1
