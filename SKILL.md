---
name: podcast-skill
description: Build a searchable knowledge base from a podcast. Given a podcast name or Spotify link, list all its episodes from the RSS feed and download the transcripts from Spotify. Use when the user wants to collect or transcribe a podcast's episodes. (Steps 3-4, RAG index and Q&A agent, are not built yet.)
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
- Other transcript sources can be added later as providers (`src/podcast_skill/providers/`) and listed under `transcribe.providers` in `config.yaml`.

## Not built yet
Step 3 (FAISS index) and step 4 (question-answering agent). `src/rag_agent.py` and `config_podcast.yaml` are the old prototype and are not used by steps 1-2.

## Tests
`.venv/bin/python -m pytest` (offline; network and browser are faked).
