import io
import sys

from podcast_skill.terminal import RtlFixStream, fix_rtl_output, visual_order


def test_hebrew_is_reversed_to_visual_order():
    assert visual_order("שלום עולם") == "םלוע םולש"


def test_brackets_are_mirrored_and_numbers_keep_their_order():
    assert visual_order("#3 אמא אמא (2018)") == "(2018) אמא אמא #3"


def test_text_without_hebrew_is_untouched():
    assert visual_order("[12:03] plain english (ok)") == "[12:03] plain english (ok)"


def test_stream_reorders_each_line_and_keeps_newlines():
    target = io.StringIO()
    stream = RtlFixStream(target)
    stream.write("שלום עולם\nenglish\n")
    stream.write("\n")
    assert target.getvalue() == "םלוע םולש\nenglish\n\n"


def test_long_line_is_wrapped_before_reordering(monkeypatch):
    monkeypatch.setattr("shutil.get_terminal_size", lambda *a, **k: __import__("os").terminal_size((21, 24)))
    target = io.StringIO()
    RtlFixStream(target).write("  אחת שתיים שלוש ארבע חמש שש שבע\n")
    rows = target.getvalue().splitlines()
    assert len(rows) > 1
    assert all(len(row) <= 20 for row in rows)
    assert rows[0].lstrip().startswith("םייתש") is False  # first row holds the START of the sentence
    assert "תחא" in rows[0]                               # "אחת" reversed


def test_stream_delegates_other_attributes():
    target = io.StringIO()
    stream = RtlFixStream(target)
    stream.write("x")
    stream.flush()
    assert stream.getvalue() == "x"


def test_not_wrapped_when_stdout_is_not_a_terminal(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())  # StringIO.isatty() is False
    assert fix_rtl_output() is False
    assert not isinstance(sys.stdout, RtlFixStream)


def test_disabled_by_config(monkeypatch):
    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdout", Tty())
    assert fix_rtl_output(enabled=False) is False
    assert fix_rtl_output(enabled=True) is True
