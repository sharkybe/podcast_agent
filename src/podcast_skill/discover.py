"""Step 1: find the podcast, read its RSS feed, write artifacts/<show>/episodes.csv."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable

from podcast_skill import episodes_csv as csvstore
from podcast_skill.config import section
from podcast_skill.episodes_csv import Episode
from podcast_skill.paths import ShowPaths, slugify
from podcast_skill.providers import spotify

ITUNES_SEARCH = "https://itunes.apple.com/search"
NS = {
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "podcast": "https://podcastindex.org/namespace/1.0",
}
USER_AGENT = "podcast-skill/0.1 (+https://github.com)"


class NeedsChoice(Exception):
    """Raised when the user must confirm a candidate (non-interactive run)."""

    def __init__(self, label: str, candidates: list[str]):
        super().__init__(f"Please choose the correct {label}")
        self.label = label
        self.candidates = candidates


@dataclass
class ShowCandidate:
    title: str
    author: str
    feed_url: str
    episode_count: int | None = None

    def describe(self) -> str:
        extra = f", {self.episode_count} episodes" if self.episode_count else ""
        return f"{self.title} - {self.author}{extra}\n      {self.feed_url}"


# --- network (monkeypatched in tests) --------------------------------------------------

def fetch_url(url: str, timeout: int = 30, attempts: int = 3) -> bytes:
    """GET with a few retries on transient errors (5xx, timeouts, connection resets)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == attempts:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
        time.sleep(2 * attempt)
    raise RuntimeError("unreachable")


# --- parsing ---------------------------------------------------------------------------

def parse_itunes_results(payload: dict[str, Any]) -> list[ShowCandidate]:
    candidates = []
    for item in payload.get("results", []):
        feed = item.get("feedUrl")
        if not feed:
            continue  # no public RSS feed: cannot be used
        candidates.append(
            ShowCandidate(
                title=item.get("collectionName") or item.get("trackName") or "",
                author=item.get("artistName") or "",
                feed_url=feed,
                episode_count=item.get("trackCount"),
            )
        )
    return candidates


def search_itunes(name: str, country: str = "US", limit: int = 5) -> list[ShowCandidate]:
    query = urllib.parse.urlencode(
        {"media": "podcast", "entity": "podcast", "term": name, "country": country, "limit": limit}
    )
    payload = json.loads(fetch_url(f"{ITUNES_SEARCH}?{query}"))
    return parse_itunes_results(payload)


def extract_episode_number(title: str, loose: bool = True) -> int | None:
    """Episode number from a title: "פרק 12" / "Episode 12". `loose` also accepts any bare number."""
    for pattern in (r"(?:episode|ep)\.?\s*#?\s*(\d+)", r"פרק\s*#?\s*(\d+)"):
        match = re.search(pattern, title, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    if not loose:
        return None
    fallback = re.search(r"\b(\d{1,4})\b", title)
    return int(fallback.group(1)) if fallback else None


def _iso_date(raw: str) -> str:
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return ""
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%d")


def parse_feed(xml_bytes: bytes) -> tuple[dict[str, str], list[Episode]]:
    """Parse RSS. Returns (show info, episodes ordered oldest -> newest with ids 1..n)."""
    root = ET.fromstring(xml_bytes)
    channel = root.find("channel")
    if channel is None:
        raise ValueError("Not an RSS feed: <channel> missing")

    info = {
        "title": (channel.findtext("title") or "").strip(),
        "author": (channel.findtext("itunes:author", namespaces=NS) or "").strip(),
        "description": (channel.findtext("description") or "").strip(),
    }

    raw_items = []
    for position, item in enumerate(channel.findall("item")):
        title = (item.findtext("title") or "").strip()
        enclosure = item.find("enclosure")
        transcript = item.find("podcast:transcript", NS)
        # The title decides the number (it is what listeners see); the feed's itunes:episode tag
        # is only a fallback, and a stray digit in a title is never taken for a number.
        found = extract_episode_number(title, loose=False)
        number = str(found) if found is not None else (item.findtext("itunes:episode", namespaces=NS) or "").strip()
        raw_items.append(
            (
                _iso_date(item.findtext("pubDate") or ""),
                position,
                Episode(
                    id=0,
                    title=title,
                    episode_number=number,
                    pub_date=_iso_date(item.findtext("pubDate") or ""),
                    duration=(item.findtext("itunes:duration", namespaces=NS) or "").strip(),
                    audio_url=(enclosure.get("url") if enclosure is not None else "") or "",
                    rss_transcript_url=(transcript.get("url") if transcript is not None else "") or "",
                ),
            )
        )

    # Feeds are normally newest-first; sort by date, keep feed order (reversed) for ties.
    raw_items.sort(key=lambda t: (t[0], -t[1]))
    episodes = [item for _, _, item in raw_items]
    csvstore.assign_ids(episodes)  # id = episode number from the title
    return info, sorted(episodes, key=lambda e: e.id)


# --- Spotify linking -------------------------------------------------------------------

def _normalize_title(title: str) -> str:
    text = re.sub(r"[^\w\s]", " ", title.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def match_spotify_episodes(
    episodes: list[Episode], spotify_episodes: list[tuple[str, str]]
) -> int:
    """Fill `spotify_url` on episodes by matching titles (then episode numbers).

    Returns the number of episodes newly linked.
    """
    by_title: dict[str, str] = {}
    by_number: dict[str, list[str]] = {}
    for title, url in spotify_episodes:
        by_title.setdefault(_normalize_title(title), url)
        number = extract_episode_number(title)
        if number is not None:
            by_number.setdefault(str(number), []).append(url)

    linked = 0
    for episode in episodes:
        if episode.spotify_url:
            continue
        url = by_title.get(_normalize_title(episode.title))
        if not url and episode.episode_number:
            candidates = by_number.get(episode.episode_number, [])
            if len(candidates) == 1:  # ambiguous numbers are not guessed
                url = candidates[0]
        if url:
            episode.spotify_url = url
            linked += 1
    return linked


# --- orchestration ---------------------------------------------------------------------

def choose(label: str, options: list[str], pick: int | None, interactive: bool) -> int:
    """Return a 0-based index. Without --pick and without a terminal, raise NeedsChoice."""
    if not options:
        raise ValueError(f"No {label} candidates found")
    if pick is not None:
        if not 1 <= pick <= len(options):
            raise ValueError(f"--pick must be between 1 and {len(options)}")
        return pick - 1
    if not interactive:
        raise NeedsChoice(label, options)
    print(f"\nFound these {label} candidates:")
    for number, option in enumerate(options, start=1):
        print(f"  {number}. {option}")
    while True:
        answer = input(f"Which one is correct? [1-{len(options)}, q to quit] ").strip().lower()
        if answer == "q":
            raise SystemExit("Aborted by user")
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1


def resolve_input(text: str) -> tuple[str, str | None]:
    """User input is a show name or a Spotify show link -> (name, spotify_show_url)."""
    if spotify.is_spotify_show_url(text):
        show_url = spotify.clean_show_url(text)
        oembed = json.loads(
            fetch_url("https://open.spotify.com/oembed?url=" + urllib.parse.quote(show_url, safe=""))
        )
        return str(oembed["title"]).strip(), show_url
    return text.strip(), None


def link_spotify(
    config: dict[str, Any],
    paths: ShowPaths,
    show_name: str,
    spotify_show_url: str | None,
    pick: int | None,
    interactive: bool,
) -> tuple[str, int]:
    """Find the show on Spotify (if needed, with user confirmation) and link episodes."""
    episodes = csvstore.read_episodes(paths.episodes_csv)
    with spotify.SpotifyBrowser(config) as browser:
        if spotify_show_url is None:
            found = spotify.search_shows(browser.page, show_name)
            index = choose("Spotify show", [f"{t}\n      {u}" for t, u in found], pick, interactive)
            spotify_show_url = found[index][1]
        print("Reading the Spotify episode list (can take a while for big shows)...")
        spotify_episodes = spotify.list_show_episodes(browser.page, spotify_show_url)
    linked = match_spotify_episodes(episodes, spotify_episodes)
    csvstore.write_episodes(paths.episodes_csv, episodes)
    return spotify_show_url, linked


def discover(
    config: dict[str, Any],
    user_input: str,
    pick: int | None = None,
    feed_url: str | None = None,
    spotify_pick: int | None = None,
    interactive: bool = True,
    link_to_spotify: bool | None = None,
) -> ShowPaths:
    cfg = section(config, "discover")
    show_name, spotify_show_url = resolve_input(user_input)

    if feed_url is None:
        candidates = search_itunes(
            show_name,
            country=str(cfg.get("search_country", "US")),
            limit=int(cfg.get("max_candidates", 5)),
        )
        index = choose("podcast", [c.describe() for c in candidates], pick, interactive)
        chosen = candidates[index]
    else:
        chosen = ShowCandidate(title=show_name, author="", feed_url=feed_url)

    info, episodes = parse_feed(fetch_url(chosen.feed_url))
    title = info["title"] or chosen.title or show_name
    paths = ShowPaths(config, slugify(title))

    if paths.episodes_csv.exists():
        episodes = csvstore.merge_with_existing(episodes, csvstore.read_episodes(paths.episodes_csv))
    csvstore.write_episodes(paths.episodes_csv, episodes)

    previous = {}
    if paths.show_json.exists():
        previous = json.loads(paths.show_json.read_text(encoding="utf-8"))
    show_record = {
        "title": title,
        "author": info["author"],
        "feed_url": chosen.feed_url,
        "spotify_show_url": spotify_show_url or previous.get("spotify_show_url", ""),
    }
    paths.show_json.write_text(json.dumps(show_record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Show: {title}\nEpisodes in feed: {len(episodes)}\nSaved: {paths.episodes_csv}")

    want_link = cfg.get("match_spotify_episodes", True) if link_to_spotify is None else link_to_spotify
    if want_link:
        known_url = show_record["spotify_show_url"] or None
        known_url, linked = link_spotify(config, paths, title, known_url, spotify_pick, interactive)
        show_record["spotify_show_url"] = known_url
        paths.show_json.write_text(json.dumps(show_record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Linked {linked} episodes to Spotify ({len(episodes) - linked} not linked).")
    return paths
