from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from podcast_skill.config import project_root, section


def slugify(name: str) -> str:
    """Filesystem-safe slug. Keeps unicode letters (e.g. Hebrew), drops punctuation."""
    text = unicodedata.normalize("NFKC", name).strip().lower()
    text = re.sub(r"[^\w]+", "-", text, flags=re.UNICODE).strip("-_")
    return text or "show"


def artifacts_dir(config: dict[str, Any]) -> Path:
    raw = section(config, "paths").get("artifacts_dir", "artifacts")
    path = Path(raw)
    return path if path.is_absolute() else project_root() / path


class ShowPaths:
    """All artifact locations for one show."""

    def __init__(self, config: dict[str, Any], show_slug: str):
        self.slug = show_slug
        self.root = artifacts_dir(config) / show_slug
        self.show_json = self.root / "show.json"
        self.episodes_csv = self.root / "episodes.csv"
        self.transcripts_dir = self.root / "transcripts"
        self.debug_dir = self.root / "debug"   # screenshots/HTML when a transcript is not found

    def transcript_file(self, episode_id: int) -> Path:
        return self.transcripts_dir / f"{episode_id:03d}.txt"
