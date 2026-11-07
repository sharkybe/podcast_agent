from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def config_path() -> Path:
    override = os.environ.get("PODCAST_CONFIG_PATH")
    if override:
        path = Path(override)
        return path if path.is_absolute() else project_root() / path
    return project_root() / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or config_path()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise RuntimeError(f"Config must be a mapping: {path}")
    return config


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, dict):
        raise RuntimeError(f"Config section '{name}' must be a mapping")
    return value
