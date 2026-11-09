---
name: podcast-skill
description: Build a searchable knowledge base from a podcast. Given a podcast name or Spotify link, list all its episodes from the RSS feed and download the transcripts from Spotify, then embed them for semantic search. Use when the user wants to collect, transcribe, index or search a podcast's episodes, or asks a question about what is said in the podcast (for example "in which episode do they talk about parenthood?").
---

# Podcast skill

Run everything from the repo root with the project venv:

```
export PYTHONPATH=src
.venv/bin/python -m podcast_skill <command> ...
```

All settings live in `config.yaml` (one section per step). Outputs go to `artifacts/<show-slug>/`.

## Step 1 - discover episodes

1. Ask the user for the podcast **name or Spotify show link**.
2. Run `discover "<name or link>"`. It searches Apple's podcast directory and prints candidates, then exits with code 2 and `CONFIRMATION NEEDED`.
3. **Show the candidates to the user and ask which one is right** (AskUserQuestion). Never pick for them.
4. Re-run with `--pick N`. This reads the RSS feed and writes `artifacts/<show>/episodes.csv` and `show.json`.
5. If only a name was given, the Spotify show is searched in a browser: it exits with code 2 again, so ask the user and re-run with `--pick N --spotify-pick M`. A Spotify link skips this question.
   A Playwright browser opens; if Spotify is logged out the user must log in once and press Enter.
- `--no-spotify` skips Spotify linking (episodes then cannot be transcribed in step 2).
- `--feed-url URL` skips the search when the user already has the RSS link.
- Re-running is safe: existing statuses and ids are kept, new episodes are appended.

## Step 2 - transcripts

`transcribe [--show SLUG] [--limit N] [--retry-unavailable]`

- Fetches each pending episode's transcript from Spotify and saves `artifacts/<show>/transcripts/<id>.txt`.
- Transcript format: one line per row; the first line of each speaker turn starts with `[m:ss]` (Spotify only gives a timestamp where the speaker changes), the following lines of that turn have none. Spotify does not say who is speaking, so there are no speaker labels.
- Episode status in `episodes.csv`: `pending` -> `done` | `unavailable` (no transcript on Spotify, or episode not linked) | `failed` (error, retried next run).
- Start with `--limit 2` to check that it works before running the whole show.
- Providers are tried in order (`transcribe.providers` in `config.yaml`, default `[spotify, ivrit]`). `ivrit` is a local fallback for episodes Spotify has no transcript for: it downloads the episode MP3 from the RSS feed and transcribes it with an ivrit.ai Whisper model (`faster-whisper`). Setup: `pip install -r requirements-ivrit.txt`; the first run downloads the model (~1.5 GB). It runs on the CPU, so expect a 45-minute episode to take roughly 15-25 minutes on Apple Silicon. Output has `[m:ss]` about every 30 s and no speaker labels.
- `--providers ivrit` runs only the local model (no browser). Episodes already marked `unavailable` need `--retry-unavailable` (or `--episode N`) to be picked up.
- More sources can be added as providers in `src/podcast_skill/providers/`.

## Step 3 - index and search

Needs the OpenAI key in the variable named by `index.api_key_env` (`OPEN_AI_API_KEY`): either in the environment or in `.env` at the repo root (gitignored, loaded automatically). Never put it in `config.yaml`.

`index [--show SLUG] [--episode ID] [--rebuild] [--model M]`

- Splits every `done` transcript into ~400-token chunks made of whole speaker turns, embeds them, and stores them in `artifacts/<show>/index/<model>.db` (SQLite).
- Incremental: only new or changed transcripts are embedded; episodes that are no longer `done` are removed. Safe to re-run, and an interrupted run resumes at the next episode.
- Default model `text-embedding-3-large`. `--model text-embedding-3-small` (or `index.embedding_model`) builds a separate, cheaper index next to it for testing. Each model has its own file, so they never mix.

`search "<question>" [--show SLUG] [-k N] [--episode ID] [--min-score X] [--model M] [--json]`

- Embeds the question, retrieves the `top_chunks` (default 30) most similar passages, and groups them by episode: a ranked list of episodes, each with its matching passages (`[m:ss]` start time, score, text). Scores are cosine similarity, not calibrated percentages.
- `--json` prints the full passage text. Use it when you need to read the passages to answer.
- To answer a broad question, run several searches with different phrasings (and synonyms, in the transcript's language), then answer from the passages and cite episode id, title and timestamp.

`passage EPISODE --from T [--to T] [--show SLUG]`

- Prints the transcript of one episode between two times (`90`, `1:30` or `1:02:03`; default span 3 minutes), as `[m:ss] text` turns. Use it to read what comes before or after a search hit.

`status [--show SLUG]`

- Shows how many episodes are `done` / `unavailable` / `failed` and what each index file holds. Run it before answering, to know how much of the show can be searched.

## Step 4 - answer questions about the podcast

There is no separate answering program: you (Claude) are the generator. When the user asks something about the podcast ("in which episodes do they talk about X?", "what do they say about Y?"), do this:

1. **Check coverage once** with `status`. Episodes without a transcript (`unavailable`) cannot be found by any search.
2. **Search several times**, with `search "<phrase>" --json -k 40`. The transcripts are in the show's language (Hebrew for the test show): write the queries in that language, and try different wordings and synonyms of the topic (for example "הפרעות אכילה", "אכילה רגשית", "בולימיה", "דיאטה"), not the user's sentence translated word for word. Broad questions need 3-5 searches; use `--episode ID` to dig into one episode.
3. **Read the passages and judge relevance yourself.** Scores are only cosine similarity and are not calibrated: a high score can still be off-topic, and vector search always returns something even when the podcast never covers the topic. Drop passages that do not address the question. If nothing relevant is left, say the podcast does not seem to cover it; do not stretch weak matches.
4. **Expand when a passage is cut off**, with `passage EPISODE --from T --to T` (a question and its answer, or the start of a story, often sit in neighbouring chunks).
5. **Answer in the user's language**, synthesising across episodes rather than listing hits: "which episodes" as a short list, "what do they say" as a combined description grouped by episode. Cite every claim as `#id title (date) [m:ss]` so the user can go and listen. Quote only short phrases; paraphrase the rest.
6. **State the limits honestly:**
   - the answer is based on the N episodes that have transcripts out of M; name the ids (or the date range) that could not be searched, and offer to transcribe them (step 2, `--providers ivrit --retry-unavailable`);
   - transcripts are automatic, so names and terms may be misspelled;
   - there are no speaker labels, so never say which host said something;
   - never present general knowledge as something the podcast said. If you add outside context, mark it clearly.

## Tests
`.venv/bin/python -m pytest` (offline; network and browser are faked).
