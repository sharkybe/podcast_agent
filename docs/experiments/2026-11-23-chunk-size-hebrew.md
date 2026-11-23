# Chunk size for Hebrew retrieval

**Date:** 2026-11-23 · **Corpus:** 149 transcripts of "אמא אבא" (snapshot before the episode relink) · **Model:** text-embedding-3-large

## Motivation

The index splits transcripts into chunks of `chunk_tokens` (default 400) with `overlap_tokens` (80).
Sizes are counted in OpenAI tokens, and Hebrew needs far more tokens per word than English. So the
assumption was that Hebrew needs bigger chunks than the default to carry enough meaning. We tested
that directly.

## Method

- **Eval set** (`eval/questions_he.jsonl`): 50 Hebrew questions, written from 45 random ~100-second
  windows of random episodes. Each question is labeled with its transcript and the time span of the
  answer. The questions paraphrase the speech rather than copy it, so a match can't come from shared
  words alone. 37 are `detail` questions (a specific moment) and 13 are `topic` questions ("which
  episode talks about X"). Windows with garbled transcription were skipped.
- **Setups:** chunk/overlap tokens of 200/40, 400/80, 600/120, 800/160, 1200/240 and 1600/320, which
  is a 20% overlap each time, plus 800/0 to isolate the effect of overlap. They use the index step's
  own chunker (whole speaker turns).
- Every setup embeds the full corpus into a throwaway cache, so the show's index is not touched.
  Search is the same as `search`: cosine similarity over all chunks, with episodes ranked by their
  best chunk.
- **Metrics:**
  - passage@k: one of the top k chunks contains the answer span.
  - episode@k: the right episode is among the top k episodes.
  - MRR: mean reciprocal rank.
  - budget@N: the answer is inside the top chunks that fit in N tokens. This compares sizes fairly
    at equal reading cost for the agent.
- **Script:** `scripts/chunk_eval.py`.

## Results

The corpus has 4.36M tokens and 928k words, which is **4.7 tokens per Hebrew word**. So 400 tokens is
only about 75 words, or about 40 seconds of speech.

| chunk/overlap | words | sec | passage@1 | passage@10 | MRR | episode@1 | episode@10 | topic ep@1 | budget@4000 | index MB |
|---|---|---|---|---|---|---|---|---|---|---|
| 200/40 | 35 | 20 | **0.38** | 0.66 | **0.48** | **0.48** | 0.86 | 0.31 | 0.72 | 280 |
| **400/80 (current)** | 75 | 40 | **0.38** | **0.74** | **0.48** | **0.48** | **0.90** | **0.46** | **0.74** | 151 |
| 600/120 | 107 | 59 | 0.26 | 0.60 | 0.38 | 0.36 | 0.74 | 0.23 | 0.54 | 108 |
| 800/160 | 143 | 74 | 0.28 | 0.64 | 0.39 | 0.34 | 0.76 | 0.31 | 0.56 | 89 |
| 800/0 | 141 | 72 | 0.22 | 0.52 | 0.32 | 0.28 | 0.72 | 0.31 | 0.48 | 82 |
| 1200/240 | 230 | 124 | 0.20 | 0.48 | 0.29 | 0.32 | 0.68 | 0.23 | 0.28 | 54 |
| 1600/320 | 316 | 172 | 0.20 | 0.48 | 0.29 | 0.24 | 0.66 | 0.08 | 0.28 | 40 |

("words" and "sec" are medians per chunk. "topic ep@1" counts the 13 topic questions only.)

## Conclusions

- **Bigger chunks are worse, not better.** 400/80 ties or beats every other setup on every metric.
  From 600 tokens up, quality drops steadily: at 1600 tokens, passage@1 halves, from 0.38 to 0.20.
  A long chunk mixes several subjects into one vector, so the specific moment is harder to match.
  This held for the broad topic questions too.
- **Overlap helps.** 800/160 beats 800/0 on nearly every metric.
- **200 vs 400** is a tie at the top. 400 has better recall at depth (passage@10 0.74 vs 0.66) and
  on topic questions, and it is half the index size. **Keep the default 400/80.**
- **Caveat:** 50 questions is small, so treat differences of a few points (one question = 0.02) as
  noise. The drop from 600 up is large and consistent. The questions come from short windows, which
  may favor small chunks somewhat, but the topic questions show the same pattern.

## Cost

Each setup embeds the whole corpus again, 4.2–4.7M tokens at $0.13/M, about $0.55–0.61 per setup.
In total that was about 32M tokens, or **about $4**, plus two partial runs that hit OpenAI's 1M
tokens/minute rate limit. The 50 question embeddings were negligible.

## Re-run

```
PYTHONPATH=src .venv/bin/python scripts/chunk_eval.py \
  --transcripts artifacts/_chunk_experiment/transcripts --questions eval/questions_he.jsonl \
  --out artifacts/_chunk_experiment --sizes 200:40,400:80,800:160
```

Raw outputs (`results.json`, `per_question.csv`) are in `artifacts/_chunk_experiment/`, which is
not in git. The transcript file names in the eval set are from the pre-relink snapshot. Map them by
`episode_title` before running against the live transcripts.
