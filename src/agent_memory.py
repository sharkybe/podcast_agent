from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


class MemoryStore:
    def __init__(self, memory_dir: Path, session_id: str):
        self.base_dir = memory_dir
        self.session_id = session_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = self.base_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.session_dir / "conversation.json"
        self.profile_path = self.base_dir / "profile.json"

    def load_history(self) -> list[dict[str, str]]:
        return _load_json(self.history_path, [])

    def save_history(self, history: list[dict[str, str]]) -> None:
        _write_json(self.history_path, history)

    def append_turn(self, user_message: str, assistant_message: str) -> None:
        history = self.load_history()
        history.extend(
            [
                {
                    "role": "user",
                    "content": user_message,
                    "timestamp": _utc_now(),
                },
                {
                    "role": "assistant",
                    "content": assistant_message,
                    "timestamp": _utc_now(),
                },
            ]
        )
        self.save_history(history)

    def load_profile(self) -> dict[str, Any]:
        return _load_json(
            self.profile_path,
            {
                "user_preferences": [],
                "important_contacts": [],
                "ongoing_topics": [],
                "assistant_improvements": [],
                "last_updated": None,
            },
        )

    def save_profile(self, profile: dict[str, Any]) -> None:
        profile["last_updated"] = _utc_now()
        _write_json(self.profile_path, profile)

    def reset(self) -> None:
        if self.history_path.exists():
            self.history_path.unlink()
        if self.profile_path.exists():
            self.profile_path.unlink()


def _format_history(history: list[dict[str, str]], history_turns: int) -> str:
    if not history:
        return "No previous conversation history."

    recent_items = history[-(history_turns * 2) :]
    lines = []
    for item in recent_items:
        role = item.get("role", "unknown")
        content = item.get("content", "")
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def _format_profile(profile: dict[str, Any], memory_items: int) -> str:
    sections = []
    for key, title in [
        ("user_preferences", "User preferences"),
        ("important_contacts", "Important contacts"),
        ("ongoing_topics", "Ongoing topics"),
        ("assistant_improvements", "Assistant improvements"),
    ]:
        values = profile.get(key, [])[:memory_items]
        if values:
            rendered = "\n".join(f"- {value}" for value in values)
            sections.append(f"{title}:\n{rendered}")

    if not sections:
        return "No long-term memory stored yet."

    return "\n\n".join(sections)


def build_model_messages(
    email_context: str,
    history: list[dict[str, str]],
    profile: dict[str, Any],
    history_turns: int,
    memory_items: int,
) -> list[dict[str, str]]:
    history_context = _format_history(history, history_turns)
    profile_context = _format_profile(profile, memory_items)

    return [
        {
            "role": "system",
            "content": (
                "You are a concise, accurate, privacy-aware email assistant. "
                "Use long-term memory only for stable preferences, recurring patterns, "
                "and open loops from prior conversations."
            ),
        },
        {
            "role": "system",
            "content": f"Long-term memory:\n{profile_context}",
        },
        {
            "role": "system",
            "content": f"Recent conversation history:\n{history_context}",
        },
        {
            "role": "user",
            "content": email_context,
        },
    ]


def _merge_unique(existing: list[str], new_items: list[str], limit: int) -> list[str]:
    merged: list[str] = []
    for item in existing + new_items:
        normalized = item.strip()
        if normalized and normalized not in merged:
            merged.append(normalized)
    return merged[:limit]


def _extract_json_payload(raw_text: str) -> str:
    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text


def update_profile_memory(
    client: OpenAI,
    model: str,
    store: MemoryStore,
    user_message: str,
    assistant_message: str,
    memory_items: int,
) -> None:
    current_profile = store.load_profile()
    memory_prompt = (
        "Extract durable memory from this interaction. Only keep information that should be "
        "remembered across future sessions. Return strict JSON with keys: "
        "user_preferences, important_contacts, ongoing_topics, assistant_improvements. "
        "Each value must be an array of short strings. If there is nothing worth keeping, return empty arrays.\n\n"
        f"Current profile:\n{json.dumps(current_profile, ensure_ascii=True)}\n\n"
        f"User message:\n{user_message}\n\n"
        f"Assistant response:\n{assistant_message}"
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "Return strict JSON only.",
            },
            {
                "role": "user",
                "content": memory_prompt,
            },
        ],
    )

    try:
        extracted = json.loads(_extract_json_payload(response.output_text))
    except json.JSONDecodeError:
        return

    updated_profile = {
        "user_preferences": _merge_unique(
            current_profile.get("user_preferences", []),
            extracted.get("user_preferences", []),
            memory_items,
        ),
        "important_contacts": _merge_unique(
            current_profile.get("important_contacts", []),
            extracted.get("important_contacts", []),
            memory_items,
        ),
        "ongoing_topics": _merge_unique(
            current_profile.get("ongoing_topics", []),
            extracted.get("ongoing_topics", []),
            memory_items,
        ),
        "assistant_improvements": _merge_unique(
            current_profile.get("assistant_improvements", []),
            extracted.get("assistant_improvements", []),
            memory_items,
        ),
    }
    store.save_profile(updated_profile)