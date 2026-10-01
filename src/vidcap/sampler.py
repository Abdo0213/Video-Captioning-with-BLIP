"""Frame sampling module supporting uniform, scene-based, and hybrid modes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from vidcap.config import Config
from vidcap.probe import VideoMetadata, probe_video

logger = logging.getLogger(__name__)


@dataclass
class FrameSample:
    frame_idx: int
    timestamp_sec: float
    image: Image.Image
    is_keyframe: bool = False


def resize_frame_to_max_side(img: Image.Image, max_side: int) -> Image.Image:
    """Resize PIL Image maintaining aspect ratio so longest side <= max_side."""
    if max_side <= 0:
        return img
    width, height = img.size
    longest = max(width, height)
    if longest <= max_side:
        return img

    scale = max_side / float(longest)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def detect_scene_boundaries(video_path: str, threshold: float = 27.0) -> List[Tuple[float, float]]:
    """Detect scenes using PySceneDetect ContentDetector, returning list of (start_sec, end_sec)."""
    try:
        from scenedetect import ContentDetector, detect

        scene_list = detect(video_path, ContentDetector(threshold=threshold))
        scenes = []
        for start_tc, end_tc in scene_list:
            scenes.append((start_tc.get_seconds(), end_tc.get_seconds()))
        logger.info(f"Detected {len(scenes)} scenes in {Path(video_path).name}")
        return scenes
    except Exception as e:
        logger.warning(f"Scene detection failed or scenedetect not available ({e}); falling back.")
        return []


def compute_sample_timestamps(
    metadata: VideoMetadata,
    config: Config,
) -> List[Tuple[float, bool]]:
    """
    Compute timestamps to sample and whether each is a scene keyframe.
    Returns sorted list of (timestamp_sec, is_keyframe).
    """
    mode = config.sampling_mode
    duration = metadata.duration_sec
    step = max(0.1, config.sample_every_sec)

    if duration <= 0:
        duration = metadata.total_frames / metadata.fps if metadata.fps > 0 else 1.0

    selected_points: dict[float, bool] = {}

    if mode in ("scene", "hybrid"):
        scenes = detect_scene_boundaries(metadata.path, threshold=config.scene_threshold)
        for start_sec, end_sec in scenes:
            mid_sec = (start_sec + end_sec) / 2.0
            mid_sec = max(0.0, min(duration, mid_sec))
            selected_points[round(mid_sec, 3)] = True

    if mode in ("uniform", "hybrid") or not selected_points:
        if mode == "uniform" or not selected_points:
            curr = 0.0
            while curr < duration:
                selected_points[round(curr, 3)] = False
                curr += step
        elif mode == "hybrid":
            # Ensure no gap exceeds config.sample_every_sec
            sorted_kfs = sorted(selected_points.keys())
            # From 0 to first keyframe
            kfs_with_endpoints = [0.0] + sorted_kfs + [duration]
            for i in range(len(kfs_with_endpoints) - 1):
                seg_start = kfs_with_endpoints[i]
                seg_end = kfs_with_endpoints[i + 1]
                gap = seg_end - seg_start
                if gap > step:
                    num_steps = int(gap // step)
                    for s in range(1, num_steps + 1):
                        pt = round(seg_start + s * step, 3)
                        if pt < seg_end and pt not in selected_points:
                            selected_points[pt] = False

    # Sort all points by timestamp
    sorted_points = sorted(selected_points.items(), key=lambda x: x[0])

    # Enforce max_frames cap by evenly subsampling if necessary
    if config.max_frames > 0 and len(sorted_points) > config.max_frames:
        logger.info(
            f"Sampled frames ({len(sorted_points)}) exceed MAX_FRAMES ({config.max_frames}). Subsampling."
        )
        indices = np.linspace(0, len(sorted_points) - 1, config.max_frames, dtype=int)
        sorted_points = [sorted_points[idx] for idx in indices]

    return sorted_points


def sample_frames_generator(
    video_path: str | Path,
    config: Optional[Config] = None,
    metadata: Optional[VideoMetadata] = None,
    chunk_size: int = 32,
) -> Generator[List[FrameSample], None, None]:
    """
    Generator streaming FrameSample batches so the full video is never held in memory at once.
    """
    path_str = str(Path(video_path).resolve())
    cfg = config or Config()
    meta = metadata or probe_video(path_str)

    sample_plan = compute_sample_timestamps(meta, cfg)
    if not sample_plan:
        return

    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path_str}")

    current_chunk: List[FrameSample] = []

    try:
        for target_sec, is_kf in sample_plan:
            target_frame_idx = int(round(target_sec * meta.fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                # Fallback seek with time ms
                cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
                ret, frame = cap.read()

            if not ret or frame is None:
                continue

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            resized_img = resize_frame_to_max_side(pil_img, cfg.frame_max_side)

            sample = FrameSample(
                frame_idx=target_frame_idx,
                timestamp_sec=target_sec,
                image=resized_img,
                is_keyframe=is_kf,
            )
            current_chunk.append(sample)

            if len(current_chunk) >= chunk_size:
                yield current_chunk
                current_chunk = []

        if current_chunk:
            yield current_chunk
    finally:
        cap.release()


def sample_all_frames(
    video_path: str | Path,
    config: Optional[Config] = None,
    metadata: Optional[VideoMetadata] = None,
) -> List[FrameSample]:
    """Sample all frames as a single list."""
    all_frames: List[FrameSample] = []
    for chunk in sample_frames_generator(video_path, config=config, metadata=metadata):
        all_frames.extend(chunk)
    return all_frames
