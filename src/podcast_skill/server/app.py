"""The hosted answer API (build plan phase 3), for Render's free plan.

    POST /api/ask                    {"question": "..."}  -> answer + references + what is left today
    POST /api/ask/{id}/follow-up     {"question": "..."}  -> the one follow-up to that answer
    GET  /api/limits                 -> what is left today for this browser
    GET  /health                     -> for Render's health check
    GET  /                           -> placeholder until the web page (phase 4)

Limits (all per day, reset at midnight Israel time, settings in config.yaml `server:`):
- a user (random id in a cookie) gets `questions_per_day` questions, each with
  `follow_ups_per_question` follow-ups;
- an IP address gets `questions_per_ip_per_day` questions, so clearing cookies doesn't help much;
- the whole service writes at most `answers_per_day` answers with Gemini, to stay inside its free
  tier. Past that, questions still get the list of relevant episodes, without a written answer.

A question whose answer failed (Gemini busy, or the daily budget used up) doesn't count against
the user. Answers to first questions are cached by normalised question, so a repeated question
costs no Gemini call.

Run locally:  python -m podcast_skill serve        (in-memory counters)
On Render:    see render.yaml                       (counters in Render's free Postgres)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import ipaddress
import os
import re
import secrets
import sqlite3
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from podcast_skill.ask import AskResult, Asker, Turn
from podcast_skill.config import load_config, load_dotenv, section
from podcast_skill.embeddings import DEFAULT_API_KEY_ENV, OpenAIEmbedder
from podcast_skill.paths import ShowPaths, artifacts_dir
from podcast_skill.server.store import Conversation, Store, build_store

DEFAULTS: dict[str, Any] = {
    "data_dir": None,                 # show folder: episodes.csv, show.json, index/<model>[-<N>d].db
    "index_file": None,               # pick one index file in data_dir/index/ (default: the smallest)
    "questions_per_day": 10,
    "follow_ups_per_question": 1,
    "questions_per_ip_per_day": 30,
    "answers_per_day": 300,           # Gemini answers for the whole service (questions + follow-ups)
    "max_question_chars": 300,
    "timezone": "Asia/Jerusalem",
    "count_cached": True,             # a cached answer still counts as one of the user's questions
    "trust_proxy": True,              # take the client IP from X-Forwarded-For (Render sets it)
    "cookie_name": "pa_uid",
    "keep_days": 3,                   # counters and follow-up history are dropped after this
}

NO_ANSWER = "(no answer was given)"


def server_settings(config: dict[str, Any]) -> dict[str, Any]:
    cfg = section(config, "server")
    settings = {key: cfg.get(key, default) for key, default in DEFAULTS.items()}
    if os.environ.get("PODCAST_DATA_DIR"):
        settings["data_dir"] = os.environ["PODCAST_DATA_DIR"]
    return settings


# ---------------------------------------------------------------- data folder


def find_data_dir(config: dict[str, Any], data_dir: Optional[str]) -> Path:
    if data_dir:
        path = Path(data_dir).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"Data folder {path} not found (server.data_dir / PODCAST_DATA_DIR)")
        return path.resolve()
    root = artifacts_dir(config)
    shows = sorted(p for p in root.iterdir() if (p / "episodes.csv").exists()) if root.is_dir() else []
    if len(shows) != 1:
        raise FileNotFoundError(
            f"Set server.data_dir or PODCAST_DATA_DIR: found {len(shows)} show folders under {root}"
        )
    return shows[0]


def index_meta(db_file: Path) -> dict[str, str]:
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    try:
        return dict(conn.execute("SELECT key, value FROM meta"))
    finally:
        conn.close()


def find_index_file(data_dir: Path, name: Optional[str]) -> Path:
    index_dir = data_dir / "index"
    if name:
        path = Path(name) if Path(name).is_absolute() else index_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Index file {path} not found")
        return path
    files = sorted(index_dir.glob("*.db"), key=lambda p: p.stat().st_size) if index_dir.is_dir() else []
    if not files:
        raise FileNotFoundError(f"No index file in {index_dir}")
    return files[0]   # the compact copy, when both are there


def embedder_for_index(config: dict[str, Any], db_file: Path) -> OpenAIEmbedder:
    """Questions must be embedded with the model and size the index was built with."""
    meta = index_meta(db_file)
    cfg = section(config, "index")
    return OpenAIEmbedder(
        model=meta.get("model") or cfg.get("embedding_model"),
        dimensions=int(meta["dimensions"]) if meta.get("dimensions") else None,
        api_key_env=cfg.get("api_key_env") or DEFAULT_API_KEY_ENV,
    )


# ---------------------------------------------------------------- helpers


def normalise_question(question: str) -> str:
    text = unicodedata.normalize("NFKC", question).lower()
    text = re.sub(r"[֑-ׇ]", "", text)          # Hebrew vowel points and cantillation
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def client_ip(request: Request, trust_proxy: bool) -> str:
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


def ip_key(ip: str, salt: str) -> str:
    """IPs are stored hashed. IPv6 is grouped by /64, since one home gets a whole /64."""
    try:
        addr = ipaddress.ip_address(ip)
        if addr.version == 6:
            ip = str(ipaddress.ip_network(f"{ip}/64", strict=False).network_address)
    except ValueError:
        pass
    return hashlib.sha256(f"{salt}|{ip}".encode()).hexdigest()[:32]


class QuestionIn(BaseModel):
    question: str


@dataclass
class Identity:
    user: str
    ip: str
    new_cookie: bool


def _error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": code, "message": message, **extra}, status_code=status)


def _public(result: AskResult) -> dict[str, Any]:
    """What the page gets. The internal error text stays in the log."""
    data = result.to_dict()
    data.pop("error", None)
    return data


PLACEHOLDER_PAGE = """<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title></head>
<body style="font-family:system-ui,sans-serif;max-width:40em;margin:3em auto;padding:0 16px">
<h1>{title}</h1>
<p>השירות פועל. דף השאלות עוד בבנייה.</p>
<p dir="ltr" style="text-align:left">The API is up: <code>POST /api/ask {{"question": "..."}}</code></p>
</body></html>
"""


# ---------------------------------------------------------------- app


def create_app(config: Optional[dict[str, Any]] = None, *, asker: Optional[Asker] = None,
               store: Optional[Store] = None, now: Optional[Callable[[], dt.datetime]] = None) -> FastAPI:
    """Build the app. Tests pass their own asker, store and clock; uvicorn calls it with none
    (`uvicorn --factory podcast_skill.server.app:create_app`)."""
    if config is None:
        load_dotenv()
        config = load_config()
    s = server_settings(config)
    tz = ZoneInfo(s["timezone"])
    clock = now or (lambda: dt.datetime.now(tz))

    show_title = "Podcast"
    if asker is None:
        data_dir = find_data_dir(config, s["data_dir"])
        db_file = find_index_file(data_dir, s["index_file"])
        paths = ShowPaths.at(data_dir)
        asker = Asker(config, paths, embedder=embedder_for_index(config, db_file), index_file=db_file)
        if paths.show_json.exists():
            import json

            show_title = json.loads(paths.show_json.read_text(encoding="utf-8")).get("title") or show_title
        print(f"Loaded {db_file.name}: {len(asker.index.rows)} chunks; {asker.coverage}", file=sys.stderr)
    store = store or build_store(os.environ.get("DATABASE_URL"))
    salt = os.environ.get("IP_HASH_SALT") or "podcast-agent"
    index_signature = hashlib.sha256(
        f"{asker.embedder.model}|{asker.embedder.dimensions}|{len(asker.index.rows)}|{asker.coverage}".encode()
    ).hexdigest()[:12]

    def today() -> dt.date:
        return clock().astimezone(tz).date()

    store.cleanup(today() - dt.timedelta(days=int(s["keep_days"])))

    app = FastAPI(title="Podcast answers", docs_url=None, redoc_url=None)
    app.state.asker, app.state.store, app.state.settings = asker, store, s

    def identify(request: Request) -> Identity:
        cookie = request.cookies.get(s["cookie_name"], "")
        new = not re.fullmatch(r"[A-Za-z0-9_-]{16,64}", cookie)
        user = secrets.token_urlsafe(18) if new else cookie
        return Identity(user, ip_key(client_ip(request, bool(s["trust_proxy"])), salt), new)

    def finish(response: Response, who: Identity) -> Response:
        if who.new_cookie:
            response.set_cookie(s["cookie_name"], who.user, max_age=365 * 24 * 3600, httponly=True,
                                samesite="lax", secure=os.environ.get("RENDER") is not None)
        return response

    def remaining(who: Identity, day: dt.date) -> int:
        left_user = int(s["questions_per_day"]) - store.count(day, "user", who.user)
        left_ip = int(s["questions_per_ip_per_day"]) - store.count(day, "ip", who.ip)
        return max(0, min(left_user, left_ip))

    def resets_at() -> str:
        tomorrow = today() + dt.timedelta(days=1)
        return dt.datetime.combine(tomorrow, dt.time(0), tz).isoformat()

    def check_question(body: QuestionIn) -> Optional[JSONResponse]:
        question = body.question.strip()
        if not question:
            return _error(400, "empty_question", "The question is empty.")
        if len(question) > int(s["max_question_chars"]):
            return _error(400, "question_too_long", f"Questions can be up to {s['max_question_chars']} characters.",
                          max_chars=int(s["max_question_chars"]))
        return None

    def run(question: str, history: Optional[Turn], use_llm: bool) -> AskResult:
        started = time.monotonic()
        result = asker.ask(question, history, use_llm=use_llm)
        status = "fallback: " + result.error if result.fallback else f"{len(result.references)} refs"
        print(f"ask ({'follow-up' if history else 'question'}, {len(question)} chars): {status}, "
              f"{time.monotonic() - started:.1f}s", file=sys.stderr)
        return result

    def fallback_reason(result: AskResult, use_llm: bool) -> Optional[str]:
        if not result.fallback:
            return None
        return "daily_budget" if not use_llm else "llm_unavailable"

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "chunks": len(asker.index.rows), "coverage": asker.coverage}

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        import html

        return PLACEHOLDER_PAGE.format(title=html.escape(show_title))

    @app.get("/api/limits")
    def limits(request: Request) -> Response:
        who = identify(request)
        day = today()
        return finish(JSONResponse({
            "questions_left": remaining(who, day),
            "questions_per_day": int(s["questions_per_day"]),
            "follow_ups_per_question": int(s["follow_ups_per_question"]),
            "max_question_chars": int(s["max_question_chars"]),
            "resets_at": resets_at(),
            "coverage": asker.coverage,
        }), who)

    @app.post("/api/ask")
    def ask(body: QuestionIn, request: Request) -> Response:
        who = identify(request)
        if (bad := check_question(body)) is not None:
            return finish(bad, who)
        question = body.question.strip()
        day = today()
        personal = [("user", who.user, int(s["questions_per_day"])),
                    ("ip", who.ip, int(s["questions_per_ip_per_day"]))]

        cache_key = f"{index_signature}|{normalise_question(question)}"
        cached = store.cache_get(cache_key)
        if cached is not None:
            if s["count_cached"] and not store.take(day, personal):
                return finish(_error(429, "daily_limit", "No questions left today.", questions_left=0,
                                     resets_at=resets_at()), who)
            conv_id = secrets.token_urlsafe(12)
            store.save_conversation(Conversation(conv_id, who.user, cached["standalone_question"],
                                                 cached["answer"] or NO_ANSWER, 0), day)
            return finish(JSONResponse({
                "id": conv_id, **cached, "question": question, "cached": True, "fallback_reason": None,
                "questions_left": remaining(who, day), "follow_ups_left": int(s["follow_ups_per_question"]),
            }), who)

        if not store.take(day, personal):
            return finish(_error(429, "daily_limit", "No questions left today.", questions_left=0,
                                 resets_at=resets_at()), who)
        budget = [("global", "answers", int(s["answers_per_day"]))]
        use_llm = store.take(day, budget)
        try:
            result = run(question, None, use_llm)
        except Exception as exc:   # embedding or index failure: nothing was answered
            store.give_back(day, personal + (budget if use_llm else []))
            print(f"ask failed: {exc!r}", file=sys.stderr)
            return finish(_error(503, "unavailable", "The service could not search right now. Try again soon."), who)
        if result.fallback:   # no written answer: the question doesn't count
            store.give_back(day, personal + (budget if use_llm else []))
        else:
            store.cache_put(cache_key, _public(result))

        conv_id = secrets.token_urlsafe(12)
        store.save_conversation(Conversation(conv_id, who.user, result.standalone_question,
                                             result.answer or NO_ANSWER, 0), day)
        return finish(JSONResponse({
            "id": conv_id, **_public(result), "cached": False, "fallback_reason": fallback_reason(result, use_llm),
            "questions_left": remaining(who, day), "follow_ups_left": int(s["follow_ups_per_question"]),
        }), who)

    @app.post("/api/ask/{conv_id}/follow-up")
    def follow_up(conv_id: str, body: QuestionIn, request: Request) -> Response:
        who = identify(request)
        if (bad := check_question(body)) is not None:
            return finish(bad, who)
        conv = store.get_conversation(conv_id)
        if conv is None or conv.user != who.user:
            return finish(_error(404, "not_found", "That question was not found. Ask it again."), who)
        limit = int(s["follow_ups_per_question"])
        if not store.take_follow_up(conv_id, who.user, limit):
            return finish(_error(429, "follow_up_used", "This question has no follow-ups left."), who)
        day = today()
        budget = [("global", "answers", int(s["answers_per_day"]))]
        use_llm = store.take(day, budget)
        try:
            result = run(body.question.strip(), Turn(conv.question, conv.answer), use_llm)
        except Exception as exc:
            store.give_back_follow_up(conv_id)
            if use_llm:
                store.give_back(day, budget)
            print(f"follow-up failed: {exc!r}", file=sys.stderr)
            return finish(_error(503, "unavailable", "The service could not search right now. Try again soon."), who)
        if result.fallback:   # no written answer: the follow-up can be asked again
            store.give_back_follow_up(conv_id)
            if use_llm:
                store.give_back(day, budget)
        after = store.get_conversation(conv_id)
        used = after.follow_ups if after else limit
        return finish(JSONResponse({
            "id": conv_id, **_public(result), "cached": False, "fallback_reason": fallback_reason(result, use_llm),
            "questions_left": remaining(who, day), "follow_ups_left": max(0, limit - used),
        }), who)

    return app
