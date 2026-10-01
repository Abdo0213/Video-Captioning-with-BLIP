"""Unit tests for temporal segmentation and caption clustering."""

import pytest
from PIL import Image

from vidcap.config import Config
from vidcap.sampler import FrameSample
from vidcap.segmenter import Segment, is_caption_similar, segment_frames


def test_is_caption_similar():
    assert is_caption_similar("A person walking on a road", "a person walking on the road", threshold=80)
    assert not is_caption_similar("A dog running in the park", "A car driving down the street", threshold=70)


def test_segment_frames_clustering_and_duration():
    dummy_img = Image.new("RGB", (32, 32))

    # Construct sequence of samples
    samples = [
        (FrameSample(frame_idx=0, timestamp_sec=0.0, image=dummy_img), "A man walking on street"),
        (FrameSample(frame_idx=30, timestamp_sec=1.0, image=dummy_img), "A man walking on the street"),
        (FrameSample(frame_idx=60, timestamp_sec=2.0, image=dummy_img), "A man walking down street"),
        # Big change / new scene
        (FrameSample(frame_idx=90, timestamp_sec=3.0, image=dummy_img, is_keyframe=True), "A red car parked in front of a house"),
        (FrameSample(frame_idx=120, timestamp_sec=4.0, image=dummy_img), "A red car parked outside"),
        (FrameSample(frame_idx=150, timestamp_sec=5.0, image=dummy_img), "A red car near the house"),
    ]

    config = Config(segment_min_sec=1.0, segment_max_sec=8.0, dedupe_similarity=80.0)
    segments = segment_frames(samples, video_duration=6.0, config=config)

    assert len(segments) == 2
    # First segment contains walking
    assert "walking" in segments[0].frame_captions[0].lower()
    # Second segment contains car
    assert "car" in segments[1].frame_captions[0].lower()

    # Verify timestamps are monotonic and non-overlapping
    assert segments[0].start_sec >= 0.0
    assert segments[0].end_sec <= segments[1].start_sec or segments[0].end_sec == segments[1].start_sec
    assert segments[1].end_sec <= 6.0


def test_segment_frames_min_sec_enforcement():
    dummy_img = Image.new("RGB", (32, 32))
    # 2 samples very close to each other with different captions
    samples = [
        (FrameSample(frame_idx=0, timestamp_sec=0.0, image=dummy_img), "A cat sitting"),
        (FrameSample(frame_idx=10, timestamp_sec=0.4, image=dummy_img), "A dog running"),
    ]

    config = Config(segment_min_sec=2.0, segment_max_sec=8.0, dedupe_similarity=80.0)
    segments = segment_frames(samples, video_duration=1.0, config=config)

    # Should merge tiny segment so we don't have sub-min-sec fragments
    assert len(segments) == 1
    assert segments[0].duration_sec >= 0.5
