"""Local transcription with an ivrit.ai Whisper model (faster-whisper).

Fallback for episodes Spotify has no transcript for: downloads the episode audio from the
RSS enclosure URL, transcribes it on this machine and returns lines in the same
`[m:ss] text` format as the Spotify provider. Nothing leaves the machine except the
audio download (and the one-time model download from Hugging Face).

`faster-whisper` is an optional dependency, imported only when the first episode is
transcribed, so Spotify-only users never need it.
"""
from __future__ import annotations

import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

from podcast_skill.config import section
from podcast_skill.episodes_csv import Episode

DEFAULT_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
TURN_SECONDS = 30  # no speaker labels: start a new `[m:ss]` block about every 30 s so chunking has anchors
USER_AGENT = "podcast_skill/0.1 (+local transcription)"


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def segments_to_lines(segments: Iterable[Any], turn_seconds: float = TURN_SECONDS) -> list[str]:
    """Whisper segments (objects with .start and .text) -> transcript lines.

    The first line of every ~`turn_seconds` block starts with `[m:ss]`; the rest are bare,
    exactly like the Spotify format.
    """
    lines: list[str] = []
    block_start: float | None = None
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        if block_start is None or segment.start - block_start >= turn_seconds:
            block_start = segment.start
            lines.append(f"[{format_timestamp(segment.start)}] {text}")
        else:
            lines.append(text)
    return lines


def download_audio(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)


class IvritTranscriptProvider:
    name = "ivrit"

    def __init__(
        self,
        config: dict[str, Any],
        audio_dir: Path | None = None,
        model_factory: Callable[..., Any] | None = None,
        downloader: Callable[[str, Path], None] = download_audio,
    ):
        cfg = section(section(config, "transcribe"), "ivrit")
        self.model_name: str = cfg.get("model") or DEFAULT_MODEL
        self.device: str = cfg.get("device") or "auto"
        self.compute_type: str = cfg.get("compute_type") or "int8"
        self.language: str = cfg.get("language") or "he"
        self.beam_size: int = int(cfg.get("beam_size") or 5)
        self.keep_audio: bool = bool(cfg.get("keep_audio", False))
        self.audio_dir = audio_dir
        self._model_factory = model_factory
        self._downloader = downloader
        self._model: Any = None

    def open(self) -> None:  # the model loads lazily in fetch(), only if an episode reaches this provider
        pass

    def close(self) -> None:
        self._model = None

    def _load_model(self) -> Any:
        if self._model is None:
            factory = self._model_factory
            if factory is None:
                try:
                    from faster_whisper import WhisperModel as factory
                except ImportError as exc:
                    raise RuntimeError(
                        "faster-whisper is not installed: pip install -r requirements-ivrit.txt"
                    ) from exc
            print(f"  loading {self.model_name} (the first run downloads ~1.5 GB)...")
            self._model = factory(self.model_name, device=self.device, compute_type=self.compute_type)
        return self._model

    def fetch(self, episode: Episode) -> list[str] | None:
        if not episode.audio_url:
            return None  # nothing to transcribe

        if self.keep_audio and self.audio_dir is not None:
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            audio_file = self.audio_dir / f"{episode.id:03d}.mp3"
            if not audio_file.exists():
                print("  downloading audio...")
                self._downloader(episode.audio_url, audio_file)
            return self._transcribe(audio_file)

        with tempfile.TemporaryDirectory(prefix="podcast_audio_") as tmp:
            audio_file = Path(tmp) / "episode.mp3"
            print("  downloading audio...")
            self._downloader(episode.audio_url, audio_file)
            return self._transcribe(audio_file)

    def _transcribe(self, audio_file: Path) -> list[str] | None:
        model = self._load_model()
        print("  transcribing (this takes a while)...")
        segments, _info = model.transcribe(
            str(audio_file),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,  # skips silence/music: faster and fewer hallucinated lines
        )
        lines = segments_to_lines(segments)  # `segments` is lazy: this is where the work happens
        return lines or None
