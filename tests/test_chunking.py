import pytest

from podcast_skill.chunking import chunk_transcript, format_time, parse_turns


def words(text: str) -> int:
    return len(text.split())


TRANSCRIPT = """[0:05] one two three
continued four five
[0:20] six seven
[1:02] eight nine ten
[1:03:07] eleven twelve"""


def test_parse_turns_reads_timestamps_and_continuation_lines():
    assert parse_turns(TRANSCRIPT) == [
        (5, ["one two three", "continued four five"]),
        (20, ["six seven"]),
        (62, ["eight nine ten"]),
        (3787, ["eleven twelve"]),
    ]


def test_text_before_first_timestamp_becomes_turn_at_zero():
    assert parse_turns("hello\n[0:10] world") == [(0, ["hello"]), (10, ["world"])]


def test_chunks_use_whole_turns_and_strip_timestamps():
    chunks = chunk_transcript(TRANSCRIPT, words, chunk_tokens=6, overlap_tokens=0)
    assert [c.text for c in chunks] == [
        "one two three\ncontinued four five",
        "six seven\neight nine ten",
        "eleven twelve",
    ]
    assert [(c.t_start, c.t_end) for c in chunks] == [(5, 5), (20, 62), (3787, 3787)]
    assert [c.index for c in chunks] == [0, 1, 2]
    assert not any("[" in c.text for c in chunks)


def test_overlap_repeats_trailing_turns():
    chunks = chunk_transcript(TRANSCRIPT, words, chunk_tokens=6, overlap_tokens=3)
    # the 3-word turn "eight nine ten" closes chunk 1 and opens chunk 2
    assert chunks[1].text == "six seven\neight nine ten"
    assert len(chunks) == 3
    assert chunks[2].text == "eight nine ten\neleven twelve"
    assert chunks[2].t_start == 62


def test_oversized_turn_is_split_by_lines():
    text = "[0:00] a a a\nb b b\nc c c"
    chunks = chunk_transcript(text, words, chunk_tokens=4, overlap_tokens=0)
    assert [c.text for c in chunks] == ["a a a", "b b b", "c c c"]
    assert all(c.t_start == 0 for c in chunks)


def test_single_long_line_stays_whole():
    chunks = chunk_transcript("[0:00] " + "w " * 50, words, chunk_tokens=10, overlap_tokens=0)
    assert len(chunks) == 1 and words(chunks[0].text) == 50


def test_empty_transcript_has_no_chunks():
    assert chunk_transcript("  \n\n", words) == []


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_transcript(TRANSCRIPT, words, chunk_tokens=5, overlap_tokens=5)


def test_format_time():
    assert format_time(5) == "0:05"
    assert format_time(62) == "1:02"
    assert format_time(3787) == "1:03:07"
