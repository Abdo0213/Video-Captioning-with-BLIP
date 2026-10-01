"""Unit tests for LLM merging, JSON validation, and error fallback."""

import json
from unittest.mock import MagicMock, patch
import pytest

from vidcap.config import Config
from vidcap.llm import OllamaCaptionMerger
from vidcap.segmenter import Segment


def test_parse_json_markdown_blocks():
    merger = OllamaCaptionMerger(Config())
    raw_markdown = """```json
    {
      "video_summary": "A man walks down a sunny street.",
      "segments": [
        {"segment_index": 0, "caption": "A man walks on the sidewalk."}
      ]
    }
    ```"""
    parsed = merger._parse_json_response(raw_markdown)
    assert parsed is not None
    assert parsed["video_summary"] == "A man walks down a sunny street."
    assert len(parsed["segments"]) == 1


def test_llm_fallback_on_missing_api_key():
    config = Config(ollama_api_key="", ollama_host="https://ollama.com")
    merger = OllamaCaptionMerger(config)

    segments = [
        Segment(
            start_sec=0.0,
            end_sec=3.0,
            frame_captions=["A dog plays in the grass", "A dog playing with a ball", "A dog playing with a ball"],
        )
    ]

    merged_segments, summary = merger.merge_segments(segments, video_duration=3.0)

    # Must fall back gracefully to most frequent caption
    assert len(merged_segments) == 1
    assert merged_segments[0].final_caption == "A dog playing with a ball"
    assert "video" in summary.lower()


@patch.object(OllamaCaptionMerger, "_call_with_retry")
def test_llm_json_repair_and_success(mock_call):
    config = Config(ollama_api_key="fake-key")
    merger = OllamaCaptionMerger(config)

    # First call returns malformed string, second call (repair) returns valid JSON
    malformed = "Here is the caption: A cat sleeps"
    valid_json = json.dumps({
        "video_summary": "A fluffy cat naps comfortably on the couch.",
        "segments": [
            {"segment_index": 0, "caption": "A cat sleeps on a sofa."}
        ]
    })
    mock_call.side_effect = [malformed, valid_json]

    segments = [
        Segment(start_sec=0.0, end_sec=2.5, frame_captions=["A cat sleeping", "A cat on couch"])
    ]

    merged_segments, summary = merger.merge_segments(segments, video_duration=2.5)

    assert mock_call.call_count == 2
    assert summary == "A fluffy cat naps comfortably on the couch."
    assert merged_segments[0].final_caption == "A cat sleeps on a sofa."
