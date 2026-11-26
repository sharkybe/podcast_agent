# podcast_agent

Turn a podcast into something you can ask questions about.

1. **discover** - podcast name or Spotify link -> episode list from the RSS feed (`artifacts/<show>/episodes.csv`)
2. **transcribe** - transcripts from Spotify; episodes without one fall back to local Hebrew transcription with an ivrit.ai Whisper model (optional, `requirements-ivrit.txt`), otherwise they are marked `unavailable`
3. **index** - OpenAI embeddings of the transcripts in a per-show SQLite file; `search` finds the matching passages, grouped by episode
4. **answer** - done: Claude is the agent. Following [SKILL.md](SKILL.md) (step 4), it runs several searches, reads the passages and answers questions such as "which episode talks about parenthood?" with episode and timestamp citations
   - **ask** - free answers without Claude: Gemini (free tier) writes 3 Hebrew search queries, they are embedded with your OpenAI key and searched, and Gemini answers in its own words with episode and timestamp references (no transcript text is shown). Needs `GEMINI_API_KEY` in `.env`; settings in `config.yaml` under `ask`. The Claude path in SKILL.md is unchanged
   - **serve** - the same answers as a hosted API for Render's free plan (`POST /api/ask`, one follow-up per question, 10 questions a day per user, counters in Render's free Postgres). See [docs/hosting.md](docs/hosting.md)
5. **report** - runs a list of questions and writes the best passages for each to an HTML page (`artifacts/<show>/reports/`), to judge search quality by eye; can include written answers

Usage and agent instructions: see [SKILL.md](SKILL.md).

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/playwright install chromium     # only needed if Chrome is not installed
export PYTHONPATH=src
.venv/bin/python -m podcast_skill discover "show name"
.venv/bin/python -m podcast_skill transcribe --limit 2
# put OPEN_AI_API_KEY=sk-... in .env (gitignored)
.venv/bin/python -m podcast_skill index
.venv/bin/python -m podcast_skill search "in which episodes do they talk about food?"
.venv/bin/python -m podcast_skill ask "באילו פרקים מדברים על אוכל?" --follow-up "ומה אומרים שם על ילדים?" --html   # needs GEMINI_API_KEY in .env
.venv/bin/python -m podcast_skill report --file questions.txt   # HTML page in artifacts/<show>/reports/
.venv/bin/python -m pytest
```
