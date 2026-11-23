"""Compare chunk sizes for retrieval on a show's transcripts.

For every chunk size (and overlap) it chunks all transcripts with the same code the index step
uses, embeds the chunks into a throwaway cache (never the show's index), embeds the eval
questions once, and measures how often the right passage and episode come back.

Eval set: JSON lines, one question each, with the transcript file and the time span (seconds)
where the answer is said:
    {"id": 1, "question": "...", "transcript": "006.txt", "t_from": 1125, "t_to": 1184, "kind": "detail"}

A chunk "contains the answer" when one of its speaker turns starts inside [t_from, t_to] (or the
span sits inside one long turn). An episode is ranked by its best chunk, as `search` does.

Usage (needs the OpenAI key, like `index`):
    PYTHONPATH=src .venv/bin/python scripts/chunk_eval.py \
        --transcripts artifacts/<show>/transcripts --questions eval/questions_he.jsonl \
        --out artifacts/_chunk_experiment --sizes 200:40,400:80,800:160,1200:240
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from podcast_skill.chunking import chunk_transcript, parse_turns, tiktoken_counter
from podcast_skill.config import load_config, load_dotenv, section
from podcast_skill.embeddings import OpenAIEmbedder

PRICE_PER_M = {"text-embedding-3-large": 0.13, "text-embedding-3-small": 0.02}
BUDGETS = (1000, 2000, 4000, 8000)   # tokens of passages the agent reads per search
K_PASSAGES = (1, 3, 5, 10, 30)
K_EPISODES = (1, 3, 5, 10)


def parse_sizes(text: str) -> list[tuple[int, int]]:
    sizes = []
    for part in text.split(","):
        size, _, overlap = part.partition(":")
        sizes.append((int(size), int(overlap) if overlap else int(size) // 5))
    return sizes


def _embed_with_backoff(embedder: OpenAIEmbedder, batch: list[str], attempts: int = 8) -> np.ndarray:
    """Embed one batch, waiting and retrying on rate limits (a whole corpus hits the tokens-per-minute cap)."""
    for attempt in range(attempts):
        try:
            return embedder.embed(batch)
        except Exception as error:  # openai.RateLimitError; not imported so the module loads without openai
            if "429" not in str(error) and "rate" not in str(error).lower() or attempt == attempts - 1:
                raise
            time.sleep(min(60, 5 * 2 ** attempt))
    raise AssertionError("unreachable")


def embed_parallel(embedder: OpenAIEmbedder, texts: list[str], workers: int) -> np.ndarray:
    batches = [texts[i:i + embedder.batch_size] for i in range(0, len(texts), embedder.batch_size)]
    with ThreadPoolExecutor(workers) as pool:
        parts = list(pool.map(lambda b: _embed_with_backoff(embedder, b), batches))
    return np.vstack(parts)


def build_chunks(transcripts: dict[str, str], count, size: int, overlap: int) -> list[dict]:
    rows = []
    for name, text in transcripts.items():
        for c in chunk_transcript(text, count, size, overlap):
            rows.append({"file": name, "t_start": c.t_start, "t_end": c.t_end, "text": c.text,
                         "tokens": count(c.text), "words": len(c.text.split())})
    return rows


def chunk_seconds(rows: list[dict], transcripts: dict[str, str]) -> float:
    """Median speech time a chunk covers: from its first turn to the turn after its last one."""
    starts = {name: sorted({t for t, _ in parse_turns(text)}) for name, text in transcripts.items()}
    spans = []
    for r in rows:
        nxt = [t for t in starts[r["file"]] if t > r["t_end"]]
        spans.append((nxt[0] if nxt else r["t_end"] + 30) - r["t_start"])
    return float(np.median(spans))


def evaluate(rows: list[dict], matrix: np.ndarray, questions: list[dict], qvecs: np.ndarray) -> list[dict]:
    files = np.array([r["file"] for r in rows])
    t_start = np.array([r["t_start"] for r in rows])
    t_end = np.array([r["t_end"] for r in rows])
    tokens = np.array([r["tokens"] for r in rows])
    out = []
    for q, vec in zip(questions, qvecs):
        scores = matrix @ vec
        order = np.argsort(-scores)
        gold = (files == q["transcript"]) & (t_start <= q["t_to"]) & (t_end >= q["t_from"])
        gold_ranked = gold[order]
        passage_rank = int(np.argmax(gold_ranked)) + 1 if gold_ranked.any() else None
        seen, episode_rank = [], None
        for pos in order:
            name = files[pos]
            if name not in seen:
                seen.append(name)
                if name == q["transcript"]:
                    episode_rank = len(seen)
                    break
        cum = np.cumsum(tokens[order])
        budget_hit = {}
        for b in BUDGETS:
            n = max(1, int(np.searchsorted(cum, b, side="right")))
            budget_hit[b] = bool(gold_ranked[:n].any())
        out.append({"id": q["id"], "kind": q.get("kind", ""), "passage_rank": passage_rank,
                    "episode_rank": episode_rank, "budget_hit": budget_hit,
                    "top_score": float(scores[order[0]])})
    return out


def summarize(results: list[dict]) -> dict:
    n = len(results)
    def rate(pred):
        return round(sum(1 for r in results if pred(r)) / n, 3)
    s = {f"passage@{k}": rate(lambda r, k=k: r["passage_rank"] is not None and r["passage_rank"] <= k) for k in K_PASSAGES}
    s["passage_mrr"] = round(sum(1 / r["passage_rank"] for r in results if r["passage_rank"]) / n, 3)
    s.update({f"episode@{k}": rate(lambda r, k=k: r["episode_rank"] is not None and r["episode_rank"] <= k) for k in K_EPISODES})
    s["episode_mrr"] = round(sum(1 / r["episode_rank"] for r in results if r["episode_rank"]) / n, 3)
    s.update({f"budget{b}": rate(lambda r, b=b: r["budget_hit"][b]) for b in BUDGETS})
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcripts", required=True, help="Folder of NNN.txt transcripts")
    parser.add_argument("--questions", required=True, help="Eval set, JSON lines")
    parser.add_argument("--out", required=True, help="Folder for the embedding cache and results")
    parser.add_argument("--sizes", default="200:40,400:80,600:120,800:160,1200:240,1600:320",
                        help="Comma-separated chunk_tokens:overlap_tokens")
    parser.add_argument("--model", help="Embedding model (default: index.embedding_model)")
    parser.add_argument("--workers", type=int, default=2, help="Parallel embedding requests")
    args = parser.parse_args()

    load_dotenv()
    cfg = section(load_config(), "index")
    model = args.model or cfg.get("embedding_model") or "text-embedding-3-large"
    embedder = OpenAIEmbedder(model=model, dimensions=cfg.get("dimensions") or None,
                              api_key_env=cfg.get("api_key_env") or "OPENAI_API_KEY",
                              batch_size=int(cfg.get("batch_size") or 96))
    out = Path(args.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    transcripts = {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(args.transcripts).glob("*.txt"))}
    questions = [json.loads(line) for line in Path(args.questions).read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = {q["transcript"] for q in questions} - set(transcripts)
    if missing:
        raise SystemExit(f"Questions point to transcripts that are not in the folder: {sorted(missing)}")
    count = tiktoken_counter()
    corpus_tokens = sum(count(t) for t in transcripts.values())
    corpus_words = sum(len(t.split()) for t in transcripts.values())
    print(f"{len(transcripts)} transcripts, {corpus_tokens:,} tokens, {corpus_words:,} words "
          f"({corpus_tokens / corpus_words:.2f} tokens per word), {len(questions)} questions, model {model}")

    qcache = out / "cache" / f"questions-{model}.npy"
    if qcache.exists() and len(np.load(qcache)) == len(questions):
        qvecs = np.load(qcache)
    else:
        qvecs = embedder.embed([q["question"] for q in questions])
        np.save(qcache, qvecs)

    summary, per_question = [], []
    for size, overlap in parse_sizes(args.sizes):
        rows = build_chunks(transcripts, count, size, overlap)
        cache = out / "cache" / f"{model}-{size}-{overlap}.npy"
        embedded_tokens = sum(r["tokens"] for r in rows)
        started = time.time()
        if cache.exists() and len(np.load(cache)) == len(rows):
            matrix = np.load(cache)
        else:
            print(f"Embedding {len(rows):,} chunks ({embedded_tokens:,} tokens) for size {size}/{overlap}...")
            matrix = embed_parallel(embedder, [r["text"] for r in rows], args.workers)
            np.save(cache, matrix)
        results = evaluate(rows, matrix, questions, qvecs)
        entry = {
            "chunk_tokens": size, "overlap_tokens": overlap, "chunks": len(rows),
            "median_tokens": int(np.median([r["tokens"] for r in rows])),
            "median_words": int(np.median([r["words"] for r in rows])),
            "median_seconds": chunk_seconds(rows, transcripts),
            "embedded_tokens": embedded_tokens,
            "cost_usd": round(embedded_tokens / 1e6 * PRICE_PER_M.get(model, 0.13), 3),
            "index_mb": round(matrix.nbytes / 1e6, 1),
            "embed_seconds": round(time.time() - started, 1),
            **summarize(results),
        }
        for kind in sorted({r["kind"] for r in results if r["kind"]}):
            sub = [r for r in results if r["kind"] == kind]
            entry[f"{kind}_episode@1"] = summarize(sub)["episode@1"]
            entry[f"{kind}_passage@5"] = summarize(sub)["passage@5"]
        summary.append(entry)
        per_question += [{"chunk_tokens": size, "overlap_tokens": overlap, **{k: v for k, v in r.items() if k != "budget_hit"},
                          **{f"budget{b}": v for b, v in r["budget_hit"].items()}} for r in results]
        print(json.dumps(entry, ensure_ascii=False))

    meta = {"model": model, "transcripts": len(transcripts), "questions": len(questions),
            "corpus_tokens": corpus_tokens, "corpus_words": corpus_words}
    (out / "results.json").write_text(json.dumps({"meta": meta, "configs": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "per_question.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_question[0]))
        writer.writeheader()
        writer.writerows(per_question)
    print(f"Wrote {out / 'results.json'} and {out / 'per_question.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
