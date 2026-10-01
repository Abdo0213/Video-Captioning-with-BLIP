"""Unit tests for subtitle formatting and renderer path escaping."""

import pytest
import srt
from vidcap.renderer import escape_ffmpeg_path, segments_to_srt
from vidcap.segmenter import Segment


def test_escape_ffmpeg_path():
    win_path = r"C:\Users\Body\outputs\test:video.srt"
    escaped = escape_ffmpeg_path(win_path)
    assert "\\" not in escaped or "\\:" in escaped
    assert "C\\:" in escaped


def test_segments_to_srt():
    segments = [
        Segment(
            start_sec=0.0,
            end_sec=2.5,
            frame_captions=["A dog runs"],
            final_caption="A golden retriever runs through the park.",
        ),
        Segment(
            start_sec=2.5,
            end_sec=5.0,
            frame_captions=["A dog catches a ball"],
            final_caption="The dog catches a yellow tennis ball.",
        ),
    ]

    srt_str = segments_to_srt(segments)
    parsed = list(srt.parse(srt_str))

    assert len(parsed) == 2
    assert parsed[0].index == 1
    assert parsed[0].content == "A golden retriever runs through the park."
    assert parsed[0].start.total_seconds() == 0.0
    assert parsed[0].end.total_seconds() == 2.5

    assert parsed[1].index == 2
    assert parsed[1].content == "The dog catches a yellow tennis ball."
    assert parsed[1].start.total_seconds() == 2.5
    assert parsed[1].end.total_seconds() == 5.0
