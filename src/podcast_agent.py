import csv
import os
import re
from pathlib import Path

import yaml
from openai import OpenAI
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import Page, sync_playwright


def _click_if_visible(page, selectors):
    for selector in selectors:
        locator = page.locator(selector).first
        if locator.count() and locator.is_visible():
            locator.click()
            return True
    return False


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _podcast_config_path() -> Path:
    config_override = os.environ.get("PODCAST_CONFIG_PATH")
    if config_override:
        override_path = Path(config_override)
        if override_path.is_absolute():
            return override_path
        return _project_root() / override_path
    return _project_root() / "config_podcast.yaml"


def _load_podcast_config() -> dict:
    config_path = _podcast_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Podcast config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise RuntimeError(f"Podcast config must be a mapping: {config_path}")

    return config


def _config_section(config: dict, section_name: str) -> dict:
    section = config.get(section_name) or {}
    if not isinstance(section, dict):
        raise RuntimeError(f"Config section '{section_name}' must be a mapping")
    return section


def _summary_fieldnames(config: dict) -> list[str]:
    fieldnames = _config_section(config, "csv").get("fieldnames") or []
    if not fieldnames:
        raise RuntimeError("Config field 'csv.fieldnames' is required")
    return [str(field) for field in fieldnames]


def _status_value(config: dict, status_name: str) -> str:
    statuses = _config_section(config, "status_values")
    status_value = statuses.get(status_name)
    if not status_value:
        raise RuntimeError(f"Config field 'status_values.{status_name}' is required")
    return str(status_value)


def _launch_browser(playwright, config: dict):
    browser_config = _config_section(config, "browser")
    launch_kwargs = {
        "headless": bool(browser_config.get("headless", False)),
        "slow_mo": int(browser_config.get("slow_mo", 100)),
    }

    launch_args = []
    if browser_config.get("start_maximized", True):
        launch_args.append("--start-maximized")
    if launch_args:
        launch_kwargs["args"] = launch_args

    preferred_channel = browser_config.get("channel")
    if preferred_channel:
        launch_kwargs["channel"] = preferred_channel

    try:
        return playwright.chromium.launch(
            **launch_kwargs,
        )
    except Exception:
        launch_kwargs.pop("channel", None)
        return playwright.chromium.launch(**launch_kwargs)


def _clean_transcript_lines(lines):
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
                continue
        cleaned.append(text)

    deduped = []
    seen = set()
    for line in cleaned:
        if line not in seen:
            deduped.append(line)
            seen.add(line)
    return deduped


def _sort_episode_rows(rows: list[dict[str, str]]) -> None:
    rows.sort(
        key=lambda row: int(row["episode_number"]) if row["episode_number"].isdigit() else -1,
        reverse=True,
    )


def _write_episode_summary_csv(config: dict, csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _summary_fieldnames(config)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _extract_episode_number(title: str) -> int | None:
    patterns = [
        r"(?:episode|ep)\s*#?\s*(\d+)",
        r"פרק\s*#?\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, title, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    fallback = re.search(r"\b(\d{1,4})\b", title)
    if fallback:
        return int(fallback.group(1))
    return None


def _next_synthetic_episode_number(config: dict, offset: int) -> int:
    numbering = _config_section(config, "numbering")
    synthetic_base = int(numbering.get("synthetic_episode_base", 900000))
    return synthetic_base - offset


def _load_episode_summary_csv(config: dict, csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    fieldnames = _summary_fieldnames(config)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            episode_name = (raw_row.get("episode_name") or "").strip()
            episode_number_raw = (raw_row.get("episode_number") or "").strip()
            link = (raw_row.get("link") or "").strip()
            status = (raw_row.get("status") or "").strip()
            transcript_path = (raw_row.get("transcript_path") or "").strip()
            ai_answer = (raw_row.get("ai_answer") or "").strip()

            if not link:
                continue

            episode_number = _extract_episode_number(episode_name)
            if episode_number is None and episode_number_raw.isdigit():
                episode_number = int(episode_number_raw)

            rows.append(
                {
                    "episode_number": str(episode_number) if episode_number is not None else episode_number_raw,
                    "link": link,
                    "episode_name": episode_name,
                    "status": status or _status_value(config, "not_started"),
                    "transcript_path": transcript_path,
                    "ai_answer": ai_answer,
                }
            )

            for fieldname in fieldnames:
                rows[-1].setdefault(fieldname, (raw_row.get(fieldname) or "").strip())

    _sort_episode_rows(rows)
    return rows


def _load_openai_client(config: dict) -> tuple[OpenAI, str]:
    openai_config = _config_section(config, "openai")
    api_key = openai_config.get("api_key")
    if not api_key:
        raise RuntimeError("Config field 'openai.api_key' is required for analyze mode")

    model = str(openai_config.get("model") or "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    return client, model


def _ask_openai_about_transcript(client: OpenAI, model: str, transcript_text: str, question: str) -> str:
    prompt = (
        "You are given a podcast transcript and a user question. "
        "Answer only based on the transcript.\n\n"
        f"Question:\n{question}\n\n"
        f"Transcript:\n{transcript_text}"
    )
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return (response.output_text or "").strip()


def _ensure_logged_in(page: Page) -> None:
    page.goto("https://open.spotify.com", wait_until="domcontentloaded")
    try:
        login_btn = page.locator("button:has-text('Log in'), a:has-text('Log in')").first
        login_btn.wait_for(timeout=4000)
        is_logged_out = login_btn.is_visible()
    except PlaywrightTimeoutError:
        is_logged_out = False

    if is_logged_out:
        input(
            "\nSpotify is not logged in.\n"
            "Please log in (including any OTP / two-factor step) in the browser window,\n"
            "then press Enter here to continue... "
        )
        page.wait_for_timeout(2000)


def get_episode_urls(page: Page, show_url: str, limit: int | None = None) -> list[tuple[str, str]]:
    page.goto(show_url, wait_until="domcontentloaded")
    page.wait_for_timeout(2000)

    load_more_clicked = 0
    for _ in range(200):
        if limit is not None:
            loaded_count = page.locator("a[href*='/episode/']").count()
            if loaded_count >= limit:
                break

        load_more = page.locator("button:has-text('Load more episodes')").first
        if not load_more.count() or not load_more.is_visible():
            break
        load_more.scroll_into_view_if_needed()
        load_more.click()
        load_more_clicked += 1
        page.wait_for_timeout(1200)

    if load_more_clicked:
        print(f"Clicked 'Load more episodes' {load_more_clicked} times.")

    prev_count = 0
    stable_rounds = 0
    while stable_rounds < 2:
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1500)
        links = page.locator("a[href*='/episode/']").all()
        current_count = len(links)
        if limit is not None and current_count >= limit:
            break
        if current_count == prev_count:
            stable_rounds += 1
        else:
            stable_rounds = 0
        prev_count = current_count

    links = page.locator("a[href*='/episode/']").all()
    seen_urls = set()
    episodes = []
    for link in links:
        href = link.get_attribute("href") or ""
        if not href.startswith("/episode/"):
            continue
        full_url = "https://open.spotify.com" + href
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)
        title = (link.get_attribute("aria-label") or link.inner_text()).strip()
        title = title.split("\n")[0].strip()
        episodes.append((title, full_url))

        if limit is not None and len(episodes) >= limit:
            break

    return episodes


def _resolve_with_project_root(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _project_root() / path


def _path_for_csv(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_project_root()))
    except ValueError:
        return str(path.resolve())


def _default_transcript_file(output_dir: Path, episode_number: str) -> Path:
    return output_dir / f"{int(episode_number):03d}.txt"


def _normalize_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if limit < 1:
        raise ValueError("runtime.limit must be >= 1 when provided")
    return limit


def _runtime_limit(config: dict) -> int | None:
    runtime_config = _config_section(config, "runtime")
    return _normalize_limit(runtime_config.get("limit"))


def _split_status(config: dict, status: str) -> tuple[str, str]:
    transcript_status = ""
    ai_status = ""
    ai_status_values = {
        _status_value(config, "ai_done"),
        _status_value(config, "ai_failed"),
        _status_value(config, "ai_skipped"),
    }
    for part in [piece.strip() for piece in status.split(";") if piece.strip()]:
        if part in ai_status_values:
            ai_status = part
        else:
            transcript_status = part
    return transcript_status, ai_status


def _compose_status(transcript_status: str | None, ai_status: str | None) -> str:
    parts = []
    if transcript_status:
        parts.append(transcript_status)
    if ai_status:
        parts.append(ai_status)
    return "; ".join(parts)


def _set_transcript_status(config: dict, row: dict[str, str], transcript_status: str) -> None:
    _, ai_status = _split_status(config, row.get("status", ""))
    row["status"] = _compose_status(transcript_status, ai_status or None)


def _set_ai_status(config: dict, row: dict[str, str], ai_status: str) -> None:
    transcript_status, _ = _split_status(config, row.get("status", ""))
    row["status"] = _compose_status(transcript_status or None, ai_status)


def _ensure_episode_number(config: dict, row: dict[str, str], synthetic_offset: int) -> int:
    episode_number = row.get("episode_number", "").strip()
    if episode_number.isdigit():
        return int(episode_number)

    extracted = _extract_episode_number(row.get("episode_name", ""))
    if extracted is not None:
        row["episode_number"] = str(extracted)
        return extracted

    synthetic_number = _next_synthetic_episode_number(config, synthetic_offset)
    row["episode_number"] = str(synthetic_number)
    return synthetic_number


def discover_show_episodes(config: dict) -> None:
    spotify_config = _config_section(config, "spotify")
    paths_config = _config_section(config, "paths")
    summary_csv = _resolve_with_project_root(paths_config.get("summary_csv"))
    if summary_csv is None:
        raise ValueError("Config field 'paths.summary_csv' is required")
    if summary_csv.exists():
        raise FileExistsError(f"Summary CSV already exists: {summary_csv}")

    show_url = spotify_config.get("show_url")
    if not show_url:
        raise ValueError("Config field 'spotify.show_url' is required for discover mode")
    limit = _runtime_limit(config)

    browser_config = _config_section(config, "browser")
    viewport_config = _config_section(browser_config, "viewport")
    viewport = {
        "width": int(viewport_config.get("width", 1600)),
        "height": int(viewport_config.get("height", 1000)),
    }

    with sync_playwright() as p:
        browser = _launch_browser(p, config)
        context = browser.new_context(viewport=viewport)
        page = context.new_page()

        _ensure_logged_in(page)
        print("Scraping episode list (this may take a moment for large podcasts)...")
        episodes = get_episode_urls(page, show_url, limit=limit)

        context.close()
        browser.close()

    print(f"Found {len(episodes)} episodes.")

    rows = []
    synthetic_offset = 0
    for title, url in episodes:
        episode_number = _extract_episode_number(title)
        if episode_number is None:
            episode_number = _next_synthetic_episode_number(config, synthetic_offset)
            synthetic_offset += 1
            print(f"  WARNING: Could not extract episode number from title, using synthetic id {episode_number}: {title}")

        rows.append(
            {
                "episode_number": str(episode_number),
                "link": url,
                "episode_name": title,
                "status": _status_value(config, "not_started"),
                "transcript_path": "",
                "ai_answer": "",
            }
        )

    _sort_episode_rows(rows)
    _write_episode_summary_csv(config, summary_csv, rows)
    print(f"Summary CSV created: {summary_csv}")


def _open_transcript_panel(page: Page) -> bool:
    page.wait_for_timeout(1200)

    tab_locator = page.locator("a[data-testid='transcript-tab']").first
    if tab_locator.count():
        aria_current = tab_locator.get_attribute("aria-current")
        if aria_current == "page":
            return True
        try:
            tab_locator.scroll_into_view_if_needed()
            tab_locator.click(timeout=2500)
            return True
        except Exception:
            try:
                tab_locator.click(force=True, timeout=2500)
                return True
            except Exception:
                pass

    direct_selectors = [
        "#main-view a[data-testid='transcript-tab']",
        "a[data-testid='transcript-tab']",
        "a:has-text('Transcript')",
        "button:has-text('Show transcript')",
        "button:has-text('Transcript')",
        "button[aria-label*='transcript' i]",
    ]

    for _ in range(4):
        if _click_if_visible(page, direct_selectors):
            return True
        page.wait_for_timeout(700)

    menu_selectors = [
        "[data-testid='context-menu-button']",
        "button[aria-label*='more' i]",
        "button:has-text('More')",
    ]

    for _ in range(4):
        if not _click_if_visible(page, menu_selectors):
            page.wait_for_timeout(500)
            continue
        page.wait_for_timeout(600)
        if _click_if_visible(page, direct_selectors):
            return True
        page.wait_for_timeout(400)

    return False


def _extract_transcript(
    config: dict,
    page: Page,
    url: str,
    allow_manual_fallback: bool = True,
) -> tuple[list[str], str]:
    page.goto(url, wait_until="domcontentloaded")

    _click_if_visible(
        page,
        [
            "button:has-text('Accept cookies')",
            "button:has-text('Accept all')",
            "button:has-text('I agree')",
        ],
    )

    opened = _open_transcript_panel(page)
    if not opened:
        print("  WARNING: Could not open transcript panel for this episode.")
        if not allow_manual_fallback:
            return [], _status_value(config, "transcript_not_found")
        input(
            "  Please open Transcript manually in the browser, then press Enter to continue... "
        )
        page.wait_for_timeout(800)

    transcript_lines = []
    transcript_selectors = [
        "div.e-10310-nav-bar span[data-encore-id='text'][dir='auto']",
        "div.e-10310-nav-bar div.dtBADEIv6u4tWZH3 span[data-encore-id='text'][dir='auto']",
        "span[data-testid='transcript-line']",
        "[data-testid='transcript-text']",
        "div[role='dialog'] span",
    ]

    for selector in transcript_selectors:
        try:
            page.locator(selector).first.wait_for(timeout=5000)
            raw_lines = page.locator(selector).all_inner_texts()
            transcript_lines = _clean_transcript_lines(raw_lines)
            if transcript_lines:
                break
        except PlaywrightTimeoutError:
            continue

    if not transcript_lines:
        return [], _status_value(config, "transcript_empty")

    return transcript_lines, _status_value(config, "transcript_downloaded")


def extract_transcripts(config: dict) -> None:
    paths_config = _config_section(config, "paths")
    summary_csv = _resolve_with_project_root(paths_config.get("summary_csv"))
    out_path = _resolve_with_project_root(paths_config.get("transcripts_dir"))
    if summary_csv is None or out_path is None:
        raise ValueError("Config fields 'paths.summary_csv' and 'paths.transcripts_dir' are required")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Episodes CSV not found: {summary_csv}")

    rows = _load_episode_summary_csv(config, summary_csv)
    limit = _runtime_limit(config)
    if limit is not None:
        rows = rows[:limit]

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving transcripts to: {out_path}")

    browser_config = _config_section(config, "browser")
    viewport_config = _config_section(browser_config, "viewport")
    viewport = {
        "width": int(viewport_config.get("width", 1600)),
        "height": int(viewport_config.get("height", 1000)),
    }

    with sync_playwright() as p:
        browser = _launch_browser(p, config)
        context = browser.new_context(viewport=viewport)
        page = context.new_page()

        _ensure_logged_in(page)

        synthetic_offset = 0
        for idx, row in enumerate(rows, start=1):
            title = row["episode_name"]
            url = row["link"]
            episode_number = _ensure_episode_number(config, row, synthetic_offset)
            synthetic_offset += 1
            out_file = _default_transcript_file(out_path, str(episode_number))

            transcript_status, _ = _split_status(config, row.get("status", ""))
            transcript_path = _resolve_with_project_root(row.get("transcript_path"))
            existing_file = transcript_path if transcript_path and transcript_path.exists() else None

            if existing_file is None and out_file.exists():
                existing_file = out_file

            if existing_file is not None:
                row["transcript_path"] = _path_for_csv(existing_file)
                _set_transcript_status(config, row, _status_value(config, "transcript_downloaded"))
                _write_episode_summary_csv(config, summary_csv, rows)
                print(f"[{idx}/{len(rows)}] Skipping (exists): {existing_file.name}")
                continue

            if transcript_status in {
                _status_value(config, "transcript_not_found"),
                _status_value(config, "transcript_empty"),
            }:
                print(f"[{idx}/{len(rows)}] Skipping ({transcript_status.lower()}): {title}")
                continue

            print(f"[{idx}/{len(rows)}] Downloading: {title}")
            try:
                lines, transcript_result = _extract_transcript(config, page, url, allow_manual_fallback=False)
            except Exception as exc:
                _set_transcript_status(config, row, _status_value(config, "transcript_failed"))
                _write_episode_summary_csv(config, summary_csv, rows)
                print(f"  WARNING: Transcript extraction failed for '{title}': {exc}")
                continue

            if transcript_result == _status_value(config, "transcript_not_found"):
                _set_transcript_status(config, row, _status_value(config, "transcript_not_found"))
                _write_episode_summary_csv(config, summary_csv, rows)
                print(f"  WARNING: Transcript button not found for '{title}', skipping.")
                continue

            if not lines:
                _set_transcript_status(config, row, _status_value(config, "transcript_empty"))
                _write_episode_summary_csv(config, summary_csv, rows)
                print(f"  WARNING: Transcript was empty for '{title}', skipping.")
                continue

            out_file.write_text("\n".join(lines), encoding="utf-8")
            row["transcript_path"] = _path_for_csv(out_file)
            _set_transcript_status(config, row, _status_value(config, "transcript_downloaded"))
            _write_episode_summary_csv(config, summary_csv, rows)
            print(f"  Saved {len(lines)} lines -> {out_file.name}")

        context.close()
        browser.close()

    print("Done.")


def analyze_transcripts(config: dict) -> None:
    openai_config = _config_section(config, "openai")
    question = openai_config.get("question")
    if not question:
        raise ValueError("Config field 'openai.question' is required for analyze mode")

    summary_csv = _resolve_with_project_root(_config_section(config, "paths").get("summary_csv"))
    if summary_csv is None or not summary_csv.exists():
        raise FileNotFoundError(f"Episodes CSV not found: {summary_csv}")

    rows = _load_episode_summary_csv(config, summary_csv)
    limit = _runtime_limit(config)
    if limit is not None:
        rows = rows[:limit]

    client, model = _load_openai_client(config)
    print(f"OpenAI analysis is enabled with model: {model}")

    for idx, row in enumerate(rows, start=1):
        title = row["episode_name"]
        transcript_status, ai_status = _split_status(config, row.get("status", ""))

        if row.get("ai_answer"):
            if ai_status != _status_value(config, "ai_done"):
                _set_ai_status(config, row, _status_value(config, "ai_done"))
                _write_episode_summary_csv(config, summary_csv, rows)
            print(f"[{idx}/{len(rows)}] Skipping (already analyzed): {title}")
            continue

        transcript_path = _resolve_with_project_root(row.get("transcript_path"))
        if transcript_path is None:
            _set_ai_status(config, row, _status_value(config, "ai_skipped"))
            _write_episode_summary_csv(config, summary_csv, rows)
            print(f"[{idx}/{len(rows)}] Skipping (no transcript path): {title}")
            continue

        if not transcript_path.exists():
            _set_ai_status(config, row, _status_value(config, "ai_failed"))
            _write_episode_summary_csv(config, summary_csv, rows)
            print(f"[{idx}/{len(rows)}] AI failed (missing transcript file): {transcript_path}")
            continue

        if transcript_status in {
            _status_value(config, "transcript_not_found"),
            _status_value(config, "transcript_empty"),
        }:
            _set_ai_status(config, row, _status_value(config, "ai_skipped"))
            _write_episode_summary_csv(config, summary_csv, rows)
            print(f"[{idx}/{len(rows)}] Skipping ({transcript_status.lower()}): {title}")
            continue

        transcript_text = transcript_path.read_text(encoding="utf-8")
        if not transcript_text.strip():
            _set_ai_status(config, row, _status_value(config, "ai_skipped"))
            _write_episode_summary_csv(config, summary_csv, rows)
            print(f"[{idx}/{len(rows)}] Skipping (blank transcript): {title}")
            continue

        try:
            row["ai_answer"] = _ask_openai_about_transcript(
                client=client,
                model=model,
                transcript_text=transcript_text,
                question=question,
            )
            _set_ai_status(config, row, _status_value(config, "ai_done"))
            _write_episode_summary_csv(config, summary_csv, rows)
            print(f"[{idx}/{len(rows)}] Analyzed: {title}")
        except Exception as exc:
            _set_ai_status(config, row, _status_value(config, "ai_failed"))
            _write_episode_summary_csv(config, summary_csv, rows)
            print(f"[{idx}/{len(rows)}] AI failed for '{title}': {exc}")

    print("Done.")


def get_spotify_transcript(config: dict, wait_for_manual_login: bool = True):
    spotify_config = _config_section(config, "spotify")
    browser_config = _config_section(config, "browser")
    viewport_config = _config_section(browser_config, "viewport")
    episode_url = spotify_config.get("episode_url")
    if not episode_url:
        raise ValueError("Config field 'spotify.episode_url' is required for episode mode")

    viewport = {
        "width": int(viewport_config.get("width", 1600)),
        "height": int(viewport_config.get("height", 1000)),
    }

    with sync_playwright() as p:
        browser = _launch_browser(p, config)
        context = browser.new_context(viewport=viewport)
        page = context.new_page()

        if wait_for_manual_login:
            _ensure_logged_in(page)

        lines, _ = _extract_transcript(config, page, episode_url)

        context.close()
        browser.close()
        return lines


def main() -> None:
    config = _load_podcast_config()
    action = (_config_section(config, "runtime").get("action") or "").strip().lower()

    if action == "discover":
        discover_show_episodes(config)
        return

    if action == "transcripts":
        extract_transcripts(config)
        return

    if action == "analyze":
        analyze_transcripts(config)
        return

    if action == "episode":
        lines = get_spotify_transcript(
            config,
            wait_for_manual_login=bool(_config_section(config, "runtime").get("wait_for_manual_login", True)),
        )
        if not lines:
            print("No transcript lines found.")
        else:
            output_path = _resolve_with_project_root(_config_section(config, "paths").get("single_episode_output"))
            if output_path is None:
                raise ValueError("Config field 'paths.single_episode_output' is required for episode mode")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"Saved transcript to {output_path} ({len(lines)} lines).")
        return

    raise ValueError(
        "Config field 'runtime.action' must be one of: discover, transcripts, analyze, episode"
    )


if __name__ == "__main__":
    main()
