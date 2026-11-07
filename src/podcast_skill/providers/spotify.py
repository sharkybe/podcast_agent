"""Spotify (open.spotify.com) helpers driven by Playwright.

Ported from the old monolithic podcast_agent.py. Needs a logged-in browser
session; the browser profile is persistent (config browser.user_data_dir), so the user
only logs in manually once.
"""
from __future__ import annotations

import re
import time
import urllib.parse
from pathlib import Path
from typing import Any

from podcast_skill.config import project_root, section
from podcast_skill.episodes_csv import Episode

SPOTIFY_ORIGIN = "https://open.spotify.com"
SHOW_URL_RE = re.compile(r"open\.spotify\.com/show/([A-Za-z0-9]+)")

TRANSCRIPT_SELECTORS = [
    # One element per row: either a header row ("0:22" + speaker label) or a text row.
    "#transcript-panel > div",
    # Spotify hashes its CSS class names (e-10310-..., e-101040-...) and changes them
    # between releases, so rely on the stable ids/attributes first. Lines have dir="auto";
    # timestamps and speaker labels do not.
    "#transcript-panel span[data-encore-id='text'][dir='auto']",
    "[data-testid='transcript-panel'] span[data-encore-id='text'][dir='auto']",
    # legacy selectors
    "div.e-10310-nav-bar span[data-encore-id='text'][dir='auto']",
    "div.e-10310-nav-bar div.dtBADEIv6u4tWZH3 span[data-encore-id='text'][dir='auto']",
    "span[data-testid='transcript-line']",
    "[data-testid='transcript-text']",
    "div[role='dialog'] span",
]


def is_spotify_show_url(text: str) -> bool:
    return bool(SHOW_URL_RE.search(text))


def clean_show_url(text: str) -> str:
    """Normalise a pasted Spotify show link to https://open.spotify.com/show/<id>."""
    match = SHOW_URL_RE.search(text)
    if not match:
        raise ValueError(f"Not a Spotify show link: {text}")
    return f"{SPOTIFY_ORIGIN}/show/{match.group(1)}"


def clean_transcript_lines(lines: list[str]) -> list[str]:
    cleaned = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if text == "This transcript was generated automatically. Its accuracy may vary.":
            continue
        if "דובר או דוברת" in text:
            continue
        if text.count(":") == 1:
            left, right = text.split(":")
            if left.isdigit() and right.isdigit() and len(right) == 2:
                continue  # bare timestamp such as "3:07"
        cleaned.append(text)

    return cleaned


TIMESTAMP_RE = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")


def format_transcript_rows(rows: list[str]) -> list[str]:
    """Turn raw panel rows into transcript lines, with `[m:ss]` on the first line of each turn.

    Spotify shows a timestamp (plus a speaker label) only where the speaker changes, so the
    lines that follow a header belong to that turn and stay bare. The speaker label itself
    is dropped: in Hebrew Spotify never fills in the speaker number ("... מס {0}").
    """
    lines: list[str] = []
    timestamp: str | None = None
    for raw in rows:
        parts = [part.strip() for part in raw.split("\n") if part.strip()]
        if not parts:
            continue
        if TIMESTAMP_RE.fullmatch(parts[0]):  # header row
            timestamp = parts[0]
            continue
        for text in clean_transcript_lines(parts):
            lines.append(f"[{timestamp}] {text}" if timestamp else text)
            timestamp = None
    return lines


# --- browser plumbing (not unit-tested; see tests marked "live") ---------------------

def _click_if_visible(page, selectors) -> bool:
    for selector in selectors:
        locator = page.locator(selector).first
        if locator.count() and locator.is_visible():
            locator.click()
            return True
    return False


def profile_dir(config: dict[str, Any]) -> Path:
    raw = section(config, "browser").get("user_data_dir", ".browser_profile")
    path = Path(raw)
    return path if path.is_absolute() else project_root() / path


def launch_browser_context(playwright, config: dict[str, Any], viewport: dict[str, int]):
    """Launch Chrome with a persistent profile so the Spotify login survives between runs."""
    browser_config = section(config, "browser")
    kwargs: dict[str, Any] = {
        "headless": bool(browser_config.get("headless", False)),
        "slow_mo": int(browser_config.get("slow_mo", 100)),
        "viewport": viewport,
    }
    if browser_config.get("start_maximized", True):
        kwargs["args"] = ["--start-maximized"]
    channel = browser_config.get("channel")
    if channel:
        kwargs["channel"] = channel
    user_data_dir = str(profile_dir(config))
    try:
        return playwright.chromium.launch_persistent_context(user_data_dir, **kwargs)
    except Exception:
        kwargs.pop("channel", None)  # Chrome not installed: use bundled Chromium
        return playwright.chromium.launch_persistent_context(user_data_dir, **kwargs)


def ensure_logged_in(page) -> None:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    page.goto(SPOTIFY_ORIGIN, wait_until="domcontentloaded")
    try:
        login_btn = page.locator("button:has-text('Log in'), a:has-text('Log in')").first
        login_btn.wait_for(timeout=4000)
        logged_out = login_btn.is_visible()
    except PlaywrightTimeoutError:
        logged_out = False

    if logged_out:
        input(
            "\nSpotify is not logged in.\n"
            "Please log in (including any OTP / two-factor step) in the browser window,\n"
            "then press Enter here to continue... "
        )
        page.wait_for_timeout(2000)


class SpotifyBrowser:
    """Context manager owning one Playwright browser/page, logged in to Spotify."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._pw = None
        self._context = None
        self.page = None

    def __enter__(self) -> "SpotifyBrowser":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> None:
        from playwright.sync_api import sync_playwright

        viewport_config = section(section(self.config, "browser"), "viewport")
        viewport = {
            "width": int(viewport_config.get("width", 1600)),
            "height": int(viewport_config.get("height", 1000)),
        }
        self._pw = sync_playwright().start()
        self._context = launch_browser_context(self._pw, self.config, viewport)
        self.page = self._context.pages[0] if self._context.pages else self._context.new_page()
        ensure_logged_in(self.page)

    def close(self) -> None:
        for closer in (
            lambda: self._context and self._context.close(),
            lambda: self._pw and self._pw.stop(),
        ):
            try:
                closer()
            except Exception:
                pass
        self._context = self._pw = self.page = None


def search_shows(page, query: str, max_results: int = 5) -> list[tuple[str, str]]:
    """Search Spotify for podcasts by name. Returns [(title, show_url)]."""
    url = f"{SPOTIFY_ORIGIN}/search/{urllib.parse.quote(query)}/podcasts"
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_timeout(3000)
    results, seen = [], set()
    for link in page.locator("a[href^='/show/']").all():
        href = (link.get_attribute("href") or "").split("?")[0]
        if href in seen:
            continue
        title = (link.get_attribute("aria-label") or link.inner_text()).strip().split("\n")[0].strip()
        if not title:
            continue
        seen.add(href)
        results.append((title, SPOTIFY_ORIGIN + href))
        if len(results) >= max_results:
            break
    return results


def list_show_episodes(page, show_url: str) -> list[tuple[str, str]]:
    """Scrape every episode (title, url) from a Spotify show page."""
    page.goto(show_url, wait_until="domcontentloaded")
    page.wait_for_timeout(2000)

    for _ in range(200):  # click "Load more episodes" until it disappears
        load_more = page.locator("button:has-text('Load more episodes')").first
        if not load_more.count() or not load_more.is_visible():
            break
        load_more.scroll_into_view_if_needed()
        load_more.click()
        page.wait_for_timeout(1200)

    prev_count, stable_rounds = 0, 0
    while stable_rounds < 2:  # scroll until the episode count stops growing
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1500)
        current = len(page.locator("a[href*='/episode/']").all())
        stable_rounds = stable_rounds + 1 if current == prev_count else 0
        prev_count = current

    episodes, seen = [], set()
    for link in page.locator("a[href*='/episode/']").all():
        href = link.get_attribute("href") or ""
        if not href.startswith("/episode/"):
            continue
        full_url = SPOTIFY_ORIGIN + href.split("?")[0]
        if full_url in seen:
            continue
        seen.add(full_url)
        title = (link.get_attribute("aria-label") or link.inner_text()).strip().split("\n")[0].strip()
        episodes.append((title, full_url))
    return episodes


COOKIE_BUTTONS = [
    "button:has-text('Accept cookies')",
    "button:has-text('Accept all')",
    "button:has-text('I agree')",
]
# The tab is a <button> today (it used to be an <a>), so do not pin the tag name.
TRANSCRIPT_TAB = "[data-testid='transcript-tab']"


def _panel_is_open(page, timeout_ms: int = 4000) -> bool:
    try:
        page.locator("#transcript-panel").first.wait_for(state="attached", timeout=timeout_ms)
        return True
    except Exception:
        return False


def _open_transcript_panel(page) -> bool:
    # Only consider a *visible* tab: the tab bar can contain hidden copies (responsive
    # overflow), and clicking one of those waits forever or silently does nothing.
    # click() waits for the tab to render (no fixed sleep). Success means the transcript
    # panel really appeared, not just that a click happened.
    tab = page.locator(f"{TRANSCRIPT_TAB} >> visible=true").first
    try:
        tab.click(timeout=8000)
        if _panel_is_open(page):
            return True
        print("  clicked the transcript tab but the panel did not appear")
    except Exception as exc:
        print(f"  tab click failed ({type(exc).__name__}); trying fallbacks")

    # Something may be covering the tab (cookie banner): dismiss it and try again.
    _click_if_visible(page, COOKIE_BUTTONS)
    try:
        tab.click(timeout=2500)
        if _panel_is_open(page):
            return True
    except Exception:
        pass

    direct = [
        "#main-view [data-testid='transcript-tab']",
        "a:has-text('Transcript')",
        "button:has-text('Show transcript')",
        "button:has-text('Transcript')",
        "button[aria-label*='transcript' i]",
    ]
    for _ in range(4):
        if _click_if_visible(page, direct) and _panel_is_open(page, 2000):
            return True
        page.wait_for_timeout(700)

    menu = [
        "[data-testid='context-menu-button']",
        "button[aria-label*='more' i]",
        "button:has-text('More')",
    ]
    for _ in range(4):
        if not _click_if_visible(page, menu):
            page.wait_for_timeout(500)
            continue
        page.wait_for_timeout(600)
        if _click_if_visible(page, direct) and _panel_is_open(page, 2000):
            return True
        page.wait_for_timeout(400)
    return False


_JS_READ = "(sel) => Array.from(document.querySelectorAll(sel)).map(e => (e.innerText || '').trim())"
_JS_SCROLL_TO = (
    "([sel, i]) => { const els = document.querySelectorAll(sel);"
    " if (els[i]) els[i].scrollIntoView({block: 'start'}); }"
)


_JS_LINE_HAS_TEXT = (
    "([sel, i]) => { const e = document.querySelectorAll(sel)[i];"
    " return !!e && (e.innerText || '').trim().length > 0; }"
)


def collect_transcript_lines(
    page, selector: str, max_rounds: int = 600, render_timeout_ms: int = 1500
) -> list[str]:
    """Read every transcript line, even though Spotify renders them lazily.

    The transcript panel contains one element per line, but only the lines near the
    viewport have text; the rest are empty placeholders. We keep the text by position and
    keep scrolling to the first line we have not seen until nothing new appears. After each
    scroll we wait until that line actually gets text (no fixed sleep), up to
    `render_timeout_ms`; a line that never renders counts as "no progress".
    """
    seen: dict[int, str] = {}
    stale = 0
    total = 0
    for _ in range(max_rounds):
        texts = page.evaluate(_JS_READ, selector)
        total = len(texts)
        before = len(seen)
        for index, text in enumerate(texts):
            if text and index not in seen:
                seen[index] = text
        missing = [i for i in range(total) if i not in seen]
        if not missing:
            break
        stale = stale + 1 if len(seen) == before else 0
        if stale >= 3:
            break
        page.evaluate(_JS_SCROLL_TO, [selector, missing[0]])
        try:
            page.wait_for_function(_JS_LINE_HAS_TEXT, arg=[selector, missing[0]], timeout=render_timeout_ms)
        except Exception:  # line did not render in time; the stale counter decides when to stop
            pass

    missing_count = total - len(seen)
    if seen and missing_count:
        print(f"  WARNING: {missing_count} of {total} transcript lines stayed empty and were skipped")
    return [seen[i] for i in sorted(seen)]


def _save_debug(page, debug_dir: Path | None, name: str, reason: str) -> None:
    if debug_dir is None:
        return
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(debug_dir / f"{name}.png"), full_page=True)
        (debug_dir / f"{name}.html").write_text(page.content(), encoding="utf-8")
        (debug_dir / f"{name}.txt").write_text(f"{reason}\nurl: {page.url}\n", encoding="utf-8")
        print(f"  debug files saved: {debug_dir / name}.[png|html|txt]")
    except Exception as exc:
        print(f"  (could not save debug files: {exc})")


def extract_transcript(page, url: str, debug_dir: Path | None = None, debug_name: str = "episode") -> list[str] | None:
    """Open an episode page and return transcript lines, or None if unavailable.

    When nothing is found, the reason is printed and (if `debug_dir` is given) a
    screenshot and the page HTML are saved there so the selectors can be fixed.
    """
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    started = time.monotonic()
    page.goto(url, wait_until="domcontentloaded")
    print(f"  page loaded ({time.monotonic() - started:.1f}s)")
    if not _open_transcript_panel(page):
        reason = "transcript tab/button not found on the episode page"
        print(f"  {reason}")
        _save_debug(page, debug_dir, debug_name, reason)
        return None

    for selector in TRANSCRIPT_SELECTORS:
        try:
            page.locator(selector).first.wait_for(state="attached", timeout=5000)
        except PlaywrightTimeoutError:
            continue
        print(f"  transcript tab open ({time.monotonic() - started:.1f}s), reading lines...")
        lines = format_transcript_rows(collect_transcript_lines(page, selector))
        if lines:
            print(f"  read {len(lines)} lines ({time.monotonic() - started:.1f}s total)")
            return lines
    reason = "transcript panel opened but no transcript lines matched any selector"
    print(f"  {reason}")
    _save_debug(page, debug_dir, debug_name, reason)
    return None


class SpotifyTranscriptProvider:
    name = "spotify"

    def __init__(self, config: dict[str, Any], debug_dir: Path | None = None):
        self._browser = SpotifyBrowser(config)
        self.debug_dir = debug_dir

    def open(self) -> None:
        self._browser.open()

    def close(self) -> None:
        self._browser.close()

    def fetch(self, episode: Episode) -> list[str] | None:
        if not episode.spotify_url:
            return None
        return extract_transcript(
            self._browser.page,
            episode.spotify_url,
            debug_dir=self.debug_dir,
            debug_name=f"{episode.id:03d}",
        )
