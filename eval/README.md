# Chunk-size eval (Hebrew)

`questions_he.jsonl`: 50 Hebrew questions, each labeled with the transcript file, episode title and the
time span where the answer is said (37 `detail`, 13 `topic`). Written as paraphrases from random
windows of the transcripts, so a hit can't come from shared words alone. File names are from the
2026-11-23 snapshot (149 transcripts, before the episode relink); use `episode_title` to re-map them.

Run: see `scripts/chunk_eval.py` (embeds into a throwaway cache, never the show's index).

## Results, 2026-11-23 (text-embedding-3-large, 149 transcripts, 4.36M tokens, 4.7 tokens per Hebrew word)

| chunk/overlap | words | seconds | passage@1 | passage@10 | MRR | episode@1 | episode@10 | topic ep@1 | index MB |
|---|---|---|---|---|---|---|---|---|---|
| 200/40 | 35 | 20 | 0.38 | 0.66 | 0.48 | 0.48 | 0.86 | 0.31 | 280 |
| **400/80 (current)** | 75 | 40 | 0.38 | 0.74 | 0.48 | 0.48 | 0.90 | 0.46 | 151 |
| 600/120 | 107 | 59 | 0.26 | 0.60 | 0.38 | 0.36 | 0.74 | 0.23 | 108 |
| 800/160 | 143 | 74 | 0.28 | 0.64 | 0.39 | 0.34 | 0.76 | 0.31 | 89 |
| 800/0 | 141 | 72 | 0.22 | 0.52 | 0.32 | 0.28 | 0.72 | 0.31 | 82 |
| 1200/240 | 230 | 124 | 0.20 | 0.48 | 0.29 | 0.32 | 0.68 | 0.23 | 54 |
| 1600/320 | 316 | 172 | 0.20 | 0.48 | 0.29 | 0.24 | 0.66 | 0.08 | 40 |

Bigger chunks did not help: 400/80 ties or beats every size, and quality drops steadily from 600 up.
Overlap helps (800/160 > 800/0). Each size costs about $0.55-0.60 to embed.
