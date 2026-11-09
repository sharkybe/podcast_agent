"""Step 3: embed transcripts and search them.

Storage: one SQLite file per show and embedding model (`artifacts/<show>/index/<model>.db`),
holding every chunk with its text, position in the episode and embedding. Search is exact
brute force: all vectors are loaded into one numpy matrix and scored with a dot product.
That takes milliseconds for tens of thousands of chunks, so no vector database is needed.

Indexing is incremental: an episode is re-embedded only when its transcript (or the chunking
settings) changed, and episodes that are no longer `done` are removed from the index.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from podcast_skill import episodes_csv as csvstore
from podcast_skill.chunking import TokenCounter, chunk_transcript, format_time, tiktoken_counter
from podcast_skill.config import section
from podcast_skill.embeddings import Embedder, build_embedder
from podcast_skill.paths import ShowPaths

CHUNKER_VERSION = "1"   # bump when the chunking logic changes, to force re-embedding
DEFAULT_CHUNK_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 80
DEFAULT_TOP_CHUNKS = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS indexed_episodes (
    episode_id INTEGER PRIMARY KEY,
    content_hash TEXT NOT NULL,
    n_chunks INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    t_start INTEGER NOT NULL,
    t_end INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_episode ON chunks (episode_id);
"""


def _connect(db_file: Path) -> sqlite3.Connection:
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    conn.executescript(SCHEMA)
    return conn


def _content_hash(text: str, chunk_tokens: int, overlap_tokens: int) -> str:
    payload = f"{CHUNKER_VERSION}|{chunk_tokens}|{overlap_tokens}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_meta(conn: sqlite3.Connection, embedder: Embedder) -> None:
    expected = {"model": embedder.model, "dimensions": str(embedder.dimensions or "")}
    stored = dict(conn.execute("SELECT key, value FROM meta"))
    if not stored:
        conn.executemany("INSERT INTO meta VALUES (?, ?)", expected.items())
        conn.commit()
    elif {key: stored.get(key) for key in expected} != expected:
        raise RuntimeError(f"Index was built with {stored}, not {expected}. Use --rebuild or another --model.")


def index_show(
    config: dict[str, Any],
    paths: ShowPaths,
    embedder: Optional[Embedder] = None,
    only_ids: Optional[Sequence[int]] = None,
    rebuild: bool = False,
    count_tokens: Optional[TokenCounter] = None,
) -> dict[str, int]:
    cfg = section(config, "index")
    chunk_tokens = int(cfg.get("chunk_tokens") or DEFAULT_CHUNK_TOKENS)
    overlap_tokens = int(cfg.get("overlap_tokens") if cfg.get("overlap_tokens") is not None else DEFAULT_OVERLAP_TOKENS)
    if not paths.episodes_csv.exists():
        raise FileNotFoundError(f"{paths.episodes_csv} not found: run the discover step first")

    embedder = embedder or build_embedder(config)
    count_tokens = count_tokens or tiktoken_counter()
    db_file = paths.index_file(embedder.model, embedder.dimensions)
    if rebuild and db_file.exists():
        db_file.unlink()

    episodes = {e.id: e for e in csvstore.read_episodes(paths.episodes_csv)}
    eligible = {
        e.id: paths.root / e.transcript_path
        for e in episodes.values()
        if e.status == csvstore.DONE and e.transcript_path and (paths.root / e.transcript_path).exists()
    }

    counts = {"indexed": 0, "unchanged": 0, "removed": 0, "chunks": 0}
    conn = _connect(db_file)
    try:
        _check_meta(conn, embedder)
        known = {row[0]: row[1] for row in conn.execute("SELECT episode_id, content_hash FROM indexed_episodes")}

        for episode_id in sorted(set(known) - set(eligible)):
            with conn:
                conn.execute("DELETE FROM chunks WHERE episode_id = ?", (episode_id,))
                conn.execute("DELETE FROM indexed_episodes WHERE episode_id = ?", (episode_id,))
            counts["removed"] += 1

        wanted = [i for i in sorted(eligible) if not only_ids or i in set(only_ids)]
        texts = {i: eligible[i].read_text(encoding="utf-8") for i in wanted}
        hashes = {i: _content_hash(texts[i], chunk_tokens, overlap_tokens) for i in wanted}
        todo = [i for i in wanted if bool(only_ids) or known.get(i) != hashes[i]]
        counts["unchanged"] = len(wanted) - len(todo)
        if not todo:
            print(f"Nothing to do: {len(wanted)} episode(s) already indexed with {embedder.model}.")
            return counts

        for position, episode_id in enumerate(todo, start=1):
            chunks = chunk_transcript(texts[episode_id], count_tokens, chunk_tokens, overlap_tokens)
            print(f"[{position}/{len(todo)}] #{episode_id} {episodes[episode_id].title}: {len(chunks)} chunks")
            vectors = embedder.embed([c.text for c in chunks]) if chunks else np.zeros((0, 0), dtype=np.float32)
            with conn:  # one transaction per episode: an interrupted run resumes from the next episode
                conn.execute("DELETE FROM chunks WHERE episode_id = ?", (episode_id,))
                conn.executemany(
                    "INSERT INTO chunks (episode_id, chunk_index, t_start, t_end, text, embedding) VALUES (?,?,?,?,?,?)",
                    [
                        (episode_id, c.index, c.t_start, c.t_end, c.text, vec.astype(np.float32).tobytes())
                        for c, vec in zip(chunks, vectors)
                    ],
                )
                conn.execute(
                    "INSERT OR REPLACE INTO indexed_episodes VALUES (?,?,?)",
                    (episode_id, hashes[episode_id], len(chunks)),
                )
            counts["indexed"] += 1
            counts["chunks"] += len(chunks)
    finally:
        conn.close()

    print(f"Done: {counts['indexed']} episode(s) indexed ({counts['chunks']} chunks), "
          f"{counts['unchanged']} unchanged, {counts['removed']} removed. Index: {db_file}")
    return counts


# ---------------------------------------------------------------- search


@dataclass
class ChunkHit:
    chunk_id: int
    episode_id: int
    chunk_index: int
    t_start: int
    t_end: int
    text: str
    score: float


@dataclass
class EpisodeHit:
    episode_id: int
    title: str
    pub_date: str
    spotify_url: str
    score: float                                    # best chunk score
    passages: list[ChunkHit] = field(default_factory=list)   # in episode order

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VectorIndex:
    """All chunks of one index file, held in memory for exact search."""

    def __init__(self, db_file: Path):
        if not db_file.exists():
            raise FileNotFoundError(f"{db_file} not found: run the index step first")
        conn = sqlite3.connect(db_file)
        try:
            rows = conn.execute(
                "SELECT id, episode_id, chunk_index, t_start, t_end, text, embedding FROM chunks ORDER BY id"
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            raise RuntimeError(f"{db_file} has no chunks: run the index step first")
        self.rows = [row[:6] for row in rows]
        self.matrix = np.vstack([np.frombuffer(row[6], dtype=np.float32) for row in rows])
        self.episode_ids = np.array([row[1] for row in rows])

    def search(
        self,
        query_vector: np.ndarray,
        k: int = DEFAULT_TOP_CHUNKS,
        episode_ids: Optional[Sequence[int]] = None,
        min_score: Optional[float] = None,
    ) -> list[ChunkHit]:
        scores = self.matrix @ np.asarray(query_vector, dtype=np.float32)
        if episode_ids:
            scores = np.where(np.isin(self.episode_ids, list(episode_ids)), scores, -np.inf)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        hits = []
        for pos in top:
            score = float(scores[pos])
            if score == -np.inf or (min_score is not None and score < min_score):
                continue
            chunk_id, episode_id, chunk_index, t_start, t_end, text = self.rows[pos]
            hits.append(ChunkHit(chunk_id, episode_id, chunk_index, t_start, t_end, text, score))
        return hits


def group_by_episode(hits: list[ChunkHit], episodes: dict[int, csvstore.Episode]) -> list[EpisodeHit]:
    """Episodes ranked by their best chunk; each with its matching passages in episode order."""
    grouped: dict[int, EpisodeHit] = {}
    for hit in hits:  # hits are best-first, so the first one seen is the episode's best
        episode = episodes.get(hit.episode_id)
        entry = grouped.setdefault(
            hit.episode_id,
            EpisodeHit(
                hit.episode_id,
                episode.title if episode else f"episode {hit.episode_id}",
                episode.pub_date if episode else "",
                episode.spotify_url if episode else "",
                hit.score,
            ),
        )
        entry.passages.append(hit)
    for entry in grouped.values():
        entry.passages.sort(key=lambda p: p.t_start)
    return list(grouped.values())


def search_episodes(
    config: dict[str, Any],
    paths: ShowPaths,
    query: str,
    top_chunks: Optional[int] = None,
    episode_ids: Optional[Sequence[int]] = None,
    min_score: Optional[float] = None,
    embedder: Optional[Embedder] = None,
) -> list[EpisodeHit]:
    """Semantic search: the best-matching passages, grouped into a ranked list of episodes."""
    if not query.strip():
        raise ValueError("Query cannot be empty")
    cfg = section(config, "index")
    embedder = embedder or build_embedder(config)
    top_chunks = top_chunks or int(cfg.get("top_chunks") or DEFAULT_TOP_CHUNKS)
    if min_score is None:
        min_score = cfg.get("min_score")

    index = VectorIndex(paths.index_file(embedder.model, embedder.dimensions))
    query_vector = embedder.embed([query])[0]
    hits = index.search(query_vector, top_chunks, episode_ids, min_score)
    episodes = {e.id: e for e in csvstore.read_episodes(paths.episodes_csv)}
    return group_by_episode(hits, episodes)


def search_chunks(
    config: dict[str, Any],
    paths: ShowPaths,
    queries: Sequence[str],
    k: int = 5,
    embedder: Optional[Embedder] = None,
) -> list[list[ChunkHit]]:
    """The k best passages for each query, not grouped (one embeddings call, index loaded once)."""
    if not queries or any(not q.strip() for q in queries):
        raise ValueError("Queries cannot be empty")
    embedder = embedder or build_embedder(config)
    index = VectorIndex(paths.index_file(embedder.model, embedder.dimensions))
    vectors = embedder.embed(list(queries))
    return [index.search(vector, k) for vector in vectors]


def status_report(paths: ShowPaths) -> str:
    """Coverage of a show: episodes by transcript status, and what each index file holds."""
    episodes = csvstore.read_episodes(paths.episodes_csv)
    lines = [f"Show: {paths.slug}", f"Episodes: {len(episodes)}"]
    for status in csvstore.STATUSES:
        matching = [e for e in episodes if e.status == status]
        if matching:
            ids = ""
            if status in (csvstore.UNAVAILABLE, csvstore.FAILED):
                ids = "  (ids: " + ", ".join(str(e.id) for e in matching[:30]) + (", ..." if len(matching) > 30 else "") + ")"
            lines.append(f"  {status}: {len(matching)}{ids}")
    index_files = sorted(paths.index_dir.glob("*.db")) if paths.index_dir.exists() else []
    if not index_files:
        lines.append("Index: none yet (run `index`)")
    for db_file in index_files:
        conn = sqlite3.connect(db_file)
        try:
            n_episodes, n_chunks = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(n_chunks), 0) FROM indexed_episodes"
            ).fetchone()
        finally:
            conn.close()
        lines.append(f"Index {db_file.stem}: {n_episodes} episodes, {n_chunks} chunks")
    return "\n".join(lines)


def format_hits(results: list[EpisodeHit], snippet_chars: int = 220) -> str:
    if not results:
        return "No matching passages."
    lines: list[str] = []
    for hit in results:
        lines.append(f"#{hit.episode_id} {hit.title} ({hit.pub_date})  best score {hit.score:.3f}")
        for passage in hit.passages:
            snippet = " ".join(passage.text.split())
            if len(snippet) > snippet_chars:
                snippet = snippet[:snippet_chars].rstrip() + "..."
            lines.append(f"  [{format_time(passage.t_start)}] {passage.score:.3f}  {snippet}")
    return "\n".join(lines)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    paths = ShowPaths(config)
    query = "What is the main topic of the podcast?"
    results = search_episodes(config, paths, query)
    print(format_hits(results))