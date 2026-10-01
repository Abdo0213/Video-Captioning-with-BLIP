"""Video probing and metadata extraction."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
import cv2
import imageio_ffmpeg

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    path: str
    fps: float
    total_frames: int
    duration_sec: float
    width: int
    height: int
    has_audio: bool


def detect_audio_stream(video_path: str | Path) -> bool:
    """Check if the video file contains an audio stream using bundled ffmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe, "-i", str(video_path)]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        output = proc.stderr or ""
        # Check for Audio stream indication in ffmpeg probe output
        for line in output.splitlines():
            if "Stream #" in line and "Audio:" in line:
                return True
    except Exception as e:
        logger.warning(f"Audio stream detection failed with error: {e}")
    return False


def probe_video(video_path: str | Path) -> VideoMetadata:
    """Probe video file and return detailed metadata."""
    path_str = str(Path(video_path).resolve())
    if not Path(path_str).is_file():
        raise FileNotFoundError(f"Video file not found: {path_str}")

    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file with OpenCV: {path_str}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or fps != fps:  # NaN or zero check
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration_sec = total_frames / fps if total_frames > 0 and fps > 0 else 0.0
    finally:
        cap.release()

    has_audio = detect_audio_stream(path_str)

    metadata = VideoMetadata(
        path=path_str,
        fps=fps,
        total_frames=total_frames,
        duration_sec=duration_sec,
        width=width,
        height=height,
        has_audio=has_audio,
    )
    logger.info(
        f"Probed video: {Path(path_str).name} ({width}x{height}, {fps:.2f} fps, "
        f"{duration_sec:.2f}s, {total_frames} frames, audio={has_audio})"
    )
    return metadata
