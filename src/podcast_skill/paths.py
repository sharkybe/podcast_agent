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
        self.audio_dir = self.root / "audio"   # downloaded episode audio, only if transcribe.ivrit.keep_audio
        self.index_dir = self.root / "index"  # one SQLite file per embedding model
        self.reports_dir = self.root / "reports"   # HTML search reports

    def transcript_file(self, episode_id: int) -> Path:
        return self.transcripts_dir / f"{episode_id:03d}.txt"

    def index_file(self, model: str, dimensions: int | None = None) -> Path:
        """Separate file per model (and size), so testing another model never touches this one."""
        suffix = f"-{dimensions}d" if dimensions else ""
        return self.index_dir / f"{model}{suffix}.db"
