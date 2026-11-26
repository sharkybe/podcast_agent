# Hosting the answer API on Render (build plan phase 3)

The `ask` pipeline behind a small HTTP API, on Render's free plan, with counters in Render's free
Postgres. The web page is phase 4; until then `GET /` is a placeholder.

## API

| Method and path | Body | Returns |
| --- | --- | --- |
| `POST /api/ask` | `{"question": "..."}` | `id`, `answer` (null on fallback), `references`, `fallback`, `fallback_reason`, `cached`, `questions_left`, `follow_ups_left`, `coverage` |
| `POST /api/ask/{id}/follow-up` | `{"question": "..."}` | the same fields, for the one follow-up |
| `GET /api/limits` | | `questions_left`, `questions_per_day`, `follow_ups_per_question`, `max_question_chars`, `resets_at` |
| `GET /health` | | `{"status": "ok", ...}` for Render's health check |

References hold episode number, title, date, time and Spotify link. Transcript text is never returned.

Errors are JSON `{"error": code, "message": ...}`: `400 empty_question` / `question_too_long`,
`404 not_found` (unknown id, or another browser's question), `429 daily_limit` / `follow_up_used`,
`503 unavailable` (search failed, e.g. OpenAI down; nothing is counted).

## Limits

Set in `config.yaml` under `server:`. All reset at midnight Israel time.

- 10 questions a day per user. The user is a random id in an `HttpOnly` cookie.
- 30 questions a day per IP (IPs are stored only as salted hashes; IPv6 is grouped by /64), so
  clearing cookies doesn't give unlimited questions.
- 1 follow-up per question. Follow-ups don't count as questions.
- 300 Gemini answers a day for the whole service, to stay inside the free tier. Past that,
  questions get the list of relevant episodes without a written answer (`fallback_reason: daily_budget`)
  and don't count.
- Questions up to 300 characters.
- When Gemini fails (busy, rate limit), the result lists episodes (`fallback_reason: llm_unavailable`)
  and neither the question nor the follow-up is counted.
- Answers to first questions are cached by normalised question, so a repeated question costs no
  Gemini call. It still counts as one of the user's questions (`count_cached: true`).

## The data

The index holds transcript text, so it must not go into this public repo. Put it in a **private**
repo whose root is one show folder:

```
episodes.csv
show.json
index/text-embedding-3-large-512d.db      # the compact copy (phase 2)
```

The server picks the smallest `.db` in `index/` and embeds questions with the model and size
stored in that file, so the 512-dimension copy just works. At Render build time
`scripts/fetch_data.sh` clones that repo into `data/` with a read-only token and deletes its `.git`.

## Deploy

1. Create the private data repo (for example `sharkybe/podcast_agent_data`) and push the three
   items above to it.
2. Create a fine-grained GitHub token with read-only **Contents** access to that repo only.
3. In Render: **New > Blueprint**, pick this repo. Render reads `render.yaml` and creates the web
   service and the free Postgres. It asks for:
   - `OPEN_AI_API_KEY`: your OpenAI key (question embeddings, about $0.04 per 1,000 questions)
   - `GEMINI_API_KEY`: the free AI Studio key
   - `DATA_REPO`: `owner/name` of the data repo
   - `DATA_REPO_TOKEN`: the token from step 2
4. Open `https://<service>.onrender.com/health`. The first request after 15 idle minutes takes
   about a minute while the free instance wakes up.

Render's free Postgres expires after 30 days. Create a new one, point `DATABASE_URL` at it, and
redeploy: the app creates its tables on startup. Counters only matter for the current day, so
nothing important is lost.

To update the data, push to the data repo, then **Manual Deploy > Clear build cache & deploy**.

## Run locally

```
.venv/bin/pip install -r requirements-server.txt
PYTHONPATH=src .venv/bin/python -m podcast_skill serve            # http://127.0.0.1:8000
curl -s -X POST localhost:8000/api/ask -H 'content-type: application/json' -d '{"question": "באילו פרקים מדברים על אוכל?"}'
```

Without `DATABASE_URL` the counters live in memory. Each question uses your OpenAI key (one small
embedding request) and your Gemini key.

## Tests

`pytest` runs the API and store tests offline with fakes. To also run the store tests against a
real Postgres, set `TEST_DATABASE_URL=postgresql://...` (the tests drop and recreate their tables).
