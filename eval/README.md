# Chunk-size eval (Hebrew)

`questions_he.jsonl`: 50 Hebrew questions, each labeled with the transcript file, episode title and the
time span where the answer is said (37 `detail`, 13 `topic`). Written as paraphrases from random
windows of the transcripts, so a hit can't come from shared words alone. File names are from the
2026-11-23 snapshot (149 transcripts, before the episode relink); use `episode_title` to re-map them.

Run: see `scripts/chunk_eval.py` (embeds into a throwaway cache, never the show's index).

Results and conclusions: [docs/experiments/2026-11-23-chunk-size-hebrew.md](../docs/experiments/2026-11-23-chunk-size-hebrew.md).
