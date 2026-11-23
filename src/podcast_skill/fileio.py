"""Crash-safe file writes.

Every output that a re-run trusts (transcripts, episodes.csv) is written to a temporary
file in the same folder and then renamed over the target. A rename is atomic, so a crash or
Ctrl-C leaves either the old file or the complete new one, never a half-written file that
the next run would mistake for finished work.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator


@contextmanager
def atomic_open(path: Path, mode: str = "w", **kwargs) -> Iterator[IO]:
    """Open a temporary sibling of `path`; on success it replaces `path`, on error it is removed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, mode, **kwargs) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    with atomic_open(path, "w", encoding=encoding) as handle:
        handle.write(text)
