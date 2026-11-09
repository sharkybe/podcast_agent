"""Split a transcript into chunks for embedding.

A transcript is a list of speaker turns. The first line of a turn starts with `[m:ss]`
(or `[h:mm:ss]`); the other lines of the turn have no timestamp. Chunks are built from whole
turns, so they never start or end mid-sentence, and each chunk knows where it sits in the
episode (`t_start`, `t_end` in seconds). Timestamps are removed from the chunk text: they
carry no meaning and would only add noise to the embedding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

TIMESTAMP_RE = re.compile(r"^\[(\d+):(\d{2})(?::(\d{2}))?\]\s*")

TokenCounter = Callable[[str], int]


@dataclass
class Chunk:
    index: int
    t_start: int   # seconds: start of the first turn in the chunk
    t_end: int     # seconds: start of the last turn in the chunk
    text: str


@dataclass
class _Unit:
    t: int
    text: str
    tokens: int


def parse_timestamp(match: re.Match) -> int:
    first, second, third = match.groups()
    if third is None:  # [m:ss]
        return int(first) * 60 + int(second)
    return int(first) * 3600 + int(second) * 60 + int(third)  # [h:mm:ss]


def format_time(seconds: int) -> str:
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def parse_turns(transcript: str) -> list[tuple[int, list[str]]]:
    """Return [(start_seconds, [lines])] for each speaker turn."""
    turns: list[tuple[int, list[str]]] = []
    for raw in transcript.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = TIMESTAMP_RE.match(line)
        if match:
            turns.append((parse_timestamp(match), []))
            line = line[match.end():].strip()
        elif not turns:  # text before the first timestamp: treat as a turn at 0:00
            turns.append((0, []))
        if line:
            turns[-1][1].append(line)
    return [(t, lines) for t, lines in turns if lines]


def _units(transcript: str, count_tokens: TokenCounter, max_tokens: int) -> list[_Unit]:
    """One unit per turn; a turn longer than a whole chunk is split into its lines."""
    units: list[_Unit] = []
    for t, lines in parse_turns(transcript):
        whole = "\n".join(lines)
        whole_tokens = count_tokens(whole)
        if whole_tokens <= max_tokens or len(lines) == 1:
            units.append(_Unit(t, whole, whole_tokens))
        else:
            units.extend(_Unit(t, line, count_tokens(line)) for line in lines)
    return units


def _overlap_tail(units: list[_Unit], overlap_tokens: int) -> list[_Unit]:
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(units):
        if total + unit.tokens > overlap_tokens:
            break
        tail.insert(0, unit)
        total += unit.tokens
    return tail


def _make_chunk(index: int, units: list[_Unit]) -> Chunk:
    return Chunk(index=index, t_start=units[0].t, t_end=units[-1].t, text="\n".join(u.text for u in units))


def chunk_transcript(
    transcript: str,
    count_tokens: TokenCounter,
    chunk_tokens: int = 400,
    overlap_tokens: int = 80,
) -> list[Chunk]:
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be >= 1")
    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("overlap_tokens must be >= 0 and smaller than chunk_tokens")

    chunks: list[Chunk] = []
    current: list[_Unit] = []
    current_tokens = 0
    for unit in _units(transcript, count_tokens, chunk_tokens):
        if current and current_tokens + unit.tokens > chunk_tokens:
            chunks.append(_make_chunk(len(chunks), current))
            current = _overlap_tail(current, overlap_tokens)
            current_tokens = sum(u.tokens for u in current)
            if current_tokens + unit.tokens > chunk_tokens:  # overlap would crowd out the new unit
                current, current_tokens = [], 0
        current.append(unit)
        current_tokens += unit.tokens
    if current:  # always holds at least one unit that is not in a previous chunk
        chunks.append(_make_chunk(len(chunks), current))
    return chunks


def tiktoken_counter(encoding_name: str = "cl100k_base") -> TokenCounter:
    """Token counter matching the OpenAI embedding models (all use cl100k_base)."""
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    return lambda text: len(encoding.encode(text))
