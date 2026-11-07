"""collect_transcript_lines against a fake page that lazily renders lines, like Spotify."""
import pytest

from podcast_skill.providers.spotify import collect_transcript_lines

PAGE = """
<div id="panel" style="height:200px; overflow:auto">
  <div id="transcript-panel"></div>
</div>
<script>
  const N = 300, ROW = 30;
  const panel = document.getElementById('panel');
  const host = document.getElementById('transcript-panel');
  for (let i = 0; i < N; i++) {
    const s = document.createElement('span');
    s.setAttribute('dir', 'auto'); s.style.cssText = 'display:block;height:' + ROW + 'px';
    host.appendChild(s);
  }
  function render() {            // only rows near the viewport have text
    const first = Math.floor(panel.scrollTop / ROW), last = first + 12;
    host.querySelectorAll('span').forEach((s, i) => {
      s.textContent = (i >= first - 2 && i <= last) ? 'line ' + i : '';
    });
  }
  panel.addEventListener('scroll', render); render();
</script>
"""


def test_collects_all_lazily_rendered_lines():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=True)
        except Exception:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception:
                pytest.skip("no Chrome/Chromium available")
        page = browser.new_page()
        page.set_content(PAGE)
        lines = collect_transcript_lines(page, "#transcript-panel span[dir='auto']", render_timeout_ms=500)
        browser.close()
    assert lines == [f"line {i}" for i in range(300)]
