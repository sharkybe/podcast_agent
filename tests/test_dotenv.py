import os

from podcast_skill.config import load_dotenv


def test_load_dotenv_parses_lines_and_keeps_existing_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "\n"
        "PODCAST_T_PLAIN=abc\n"
        "export PODCAST_T_EXPORT = spaced \n"
        'PODCAST_T_QUOTED="in quotes"\n'
        "PODCAST_T_EXISTING=from-file\n"
        "not a variable line\n",
        encoding="utf-8",
    )
    for name in ("PODCAST_T_PLAIN", "PODCAST_T_EXPORT", "PODCAST_T_QUOTED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PODCAST_T_EXISTING", "from-env")

    load_dotenv(env_file)
    try:
        assert os.environ["PODCAST_T_PLAIN"] == "abc"
        assert os.environ["PODCAST_T_EXPORT"] == "spaced"
        assert os.environ["PODCAST_T_QUOTED"] == "in quotes"
        assert os.environ["PODCAST_T_EXISTING"] == "from-env"
    finally:  # load_dotenv writes os.environ directly, so undo it
        for name in ("PODCAST_T_PLAIN", "PODCAST_T_EXPORT", "PODCAST_T_QUOTED"):
            os.environ.pop(name, None)


def test_missing_env_file_is_fine(tmp_path):
    load_dotenv(tmp_path / "nope.env")
