"""Temporal segmentation and caption deduplication module."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import logging
from typing import List, Optional
from PIL import Image
from rapidfuzz import fuzz

from vidcap.config import Config
from vidcap.sampler import FrameSample

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    frame_captions: List[str] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    keyframes: List[Image.Image] = field(default_factory=list)
    final_caption: Optional[str] = None

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def most_frequent_caption(self) -> str:
        valid_caps = [c.strip() for c in self.frame_captions if c.strip()]
        if not valid_caps:
            return "Video segment"
        counts = Counter(valid_caps)
        return counts.most_common(1)[0][0]


def is_caption_similar(cap1: str, cap2: str, threshold: float = 90.0) -> bool:
    """Check if two captions are sufficiently similar using fuzzy ratio."""
    if not cap1 or not cap2:
        return False
    score = fuzz.ratio(cap1.lower(), cap2.lower())
    return score >= threshold


def segment_frames(
    captioned_samples: List[tuple[FrameSample, str]],
    video_duration: float,
    config: Optional[Config] = None,
) -> List[Segment]:
    """
    Segment frames based on scene boundaries, caption similarity, and duration constraints.
    Enforces SEGMENT_MIN_SEC and SEGMENT_MAX_SEC.
    """
    if not captioned_samples:
        return []

    cfg = config or Config()
    min_sec = cfg.segment_min_sec
    max_sec = cfg.segment_max_sec
    sim_thresh = cfg.dedupe_similarity

    # Initial clustering
    clusters: List[List[tuple[FrameSample, str]]] = []
    current_cluster: List[tuple[FrameSample, str]] = [captioned_samples[0]]

    for i in range(1, len(captioned_samples)):
        curr_sample, curr_cap = captioned_samples[i]
        prev_sample, prev_cap = current_cluster[-1]
        cluster_start_t = current_cluster[0][0].timestamp_sec
        curr_t = curr_sample.timestamp_sec

        # Decide whether to split
        split = False

        # 1. Scene boundary trigger
        if curr_sample.is_keyframe:
            split = True

        # 2. Maximum segment duration trigger
        elif (curr_t - cluster_start_t) >= max_sec:
            split = True

        # 3. Caption semantic dissimilarity trigger
        elif not is_caption_similar(prev_cap, curr_cap, sim_thresh):
            split = True

        if split:
            clusters.append(current_cluster)
            current_cluster = [(curr_sample, curr_cap)]
        else:
            current_cluster.append((curr_sample, curr_cap))

    if current_cluster:
        clusters.append(current_cluster)

    # Convert clusters to raw Segment objects with timestamp boundaries
    raw_segments: List[Segment] = []
    num_clusters = len(clusters)

    for idx, cluster in enumerate(clusters):
        t_first = cluster[0][0].timestamp_sec
        t_last = cluster[-1][0].timestamp_sec

        # Determine start time
        if idx == 0:
            start_t = 0.0
        else:
            prev_t_last = clusters[idx - 1][-1][0].timestamp_sec
            start_t = (prev_t_last + t_first) / 2.0

        # Determine end time
        if idx == num_clusters - 1:
            end_t = max(video_duration, t_last)
        else:
            next_t_first = clusters[idx + 1][0][0].timestamp_sec
            end_t = (t_last + next_t_first) / 2.0

        # Ensure start < end
        if end_t <= start_t:
            end_t = start_t + 0.5

        captions = [cap for _, cap in cluster if cap.strip()]
        timestamps = [s.timestamp_sec for s, _ in cluster]

        # Select up to 3 keyframes (middle or marked keyframes)
        keyframes: List[Image.Image] = []
        kf_samples = [s for s, _ in cluster if s.is_keyframe]
        if kf_samples:
            keyframes = [s.image for s in kf_samples[:3]]
        else:
            # Pick representative frames evenly
            if len(cluster) == 1:
                keyframes = [cluster[0][0].image]
            elif len(cluster) == 2:
                keyframes = [cluster[0][0].image, cluster[1][0].image]
            else:
                mid_idx = len(cluster) // 2
                keyframes = [cluster[0][0].image, cluster[mid_idx][0].image, cluster[-1][0].image]

        raw_segments.append(
            Segment(
                start_sec=round(start_t, 3),
                end_sec=round(end_t, 3),
                frame_captions=captions,
                timestamps=timestamps,
                keyframes=keyframes,
            )
        )

    # Enforce min_sec by merging tiny segments into neighbours
    merged_segments: List[Segment] = []
    for seg in raw_segments:
        if not merged_segments:
            merged_segments.append(seg)
            continue

        if seg.duration_sec < min_sec and len(merged_segments) > 0:
            # Merge into previous segment
            prev = merged_segments[-1]
            prev.end_sec = seg.end_sec
            prev.frame_captions.extend(seg.frame_captions)
            prev.timestamps.extend(seg.timestamps)
            for kf in seg.keyframes:
                if len(prev.keyframes) < 3:
                    prev.keyframes.append(kf)
        else:
            merged_segments.append(seg)

    # If first segment is tiny and there is a second segment, merge first into second if needed
    if len(merged_segments) > 1 and merged_segments[0].duration_sec < min_sec:
        first = merged_segments.pop(0)
        second = merged_segments[0]
        second.start_sec = first.start_sec
        second.frame_captions = first.frame_captions + second.frame_captions
        second.timestamps = first.timestamps + second.timestamps
        second.keyframes = (first.keyframes + second.keyframes)[:3]

    # Ensure non-overlapping monotonic boundaries
    for i in range(len(merged_segments)):
        if i > 0:
            if merged_segments[i].start_sec < merged_segments[i - 1].end_sec:
                merged_segments[i].start_sec = merged_segments[i - 1].end_sec
        if merged_segments[i].end_sec <= merged_segments[i].start_sec:
            merged_segments[i].end_sec = round(merged_segments[i].start_sec + 1.0, 3)

    return merged_segments
