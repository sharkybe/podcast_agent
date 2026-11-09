# podcast_agent

Turn a podcast into something you can ask questions about.

1. **discover** - podcast name or Spotify link -> episode list from the RSS feed (`artifacts/<show>/episodes.csv`)
2. **transcribe** - transcripts from Spotify; episodes without one fall back to local Hebrew transcription with an ivrit.ai Whisper model (optional, `requirements-ivrit.txt`), otherwise they are marked `unavailable`
3. **index** - OpenAI embeddings of the transcripts in a per-show SQLite file; `search` finds the matching passages, grouped by episode
4. agent (planned) - answer questions such as "which episode talks about parenthood?"

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
.venv/bin/python -m pytest
```
