from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def config(tmp_path):
    return {
        "paths": {"artifacts_dir": str(tmp_path / "artifacts")},
        "discover": {"match_spotify_episodes": False, "max_candidates": 5, "search_country": "US"},
        "transcribe": {"providers": ["spotify"], "retry_unavailable": False, "limit": None},
    }
