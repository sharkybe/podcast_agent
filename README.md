# podcast_agent

Turn a podcast into something you can ask questions about.

1. **discover** - podcast name or Spotify link -> episode list from the RSS feed (`artifacts/<show>/episodes.csv`)
2. **transcribe** - transcripts from Spotify; episodes without one are marked `unavailable`
3. index (planned) - FAISS over the transcripts
4. agent (planned) - answer questions such as "which episode talks about parenthood?"

Usage and agent instructions: see [SKILL.md](SKILL.md).

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/playwright install chromium     # only needed if Chrome is not installed
export PYTHONPATH=src
.venv/bin/python -m podcast_skill discover "show name"
.venv/bin/python -m podcast_skill transcribe --limit 2
.venv/bin/python -m pytest
```
