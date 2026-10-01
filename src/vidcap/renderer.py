"""Subtitle generation, JSON export, and video rendering / burn-in module."""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import List, Optional
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import srt

from vidcap.config import Config
from vidcap.probe import VideoMetadata, probe_video
from vidcap.segmenter import Segment

logger = logging.getLogger(__name__)


def format_timedelta(seconds: float) -> datetime.timedelta:
    """Convert float seconds to datetime.timedelta."""
    return datetime.timedelta(seconds=max(0.0, seconds))


def segments_to_srt(segments: List[Segment]) -> str:
    """Convert segments to valid SRT subtitle format."""
    subtitles = []
    for idx, seg in enumerate(segments, start=1):
        text = seg.final_caption or seg.most_frequent_caption
        text = text.strip()
        if not text:
            continue

        start_td = format_timedelta(seg.start_sec)
        end_td = format_timedelta(seg.end_sec)
        if end_td <= start_td:
            end_td = start_td + datetime.timedelta(seconds=1.0)

        subtitles.append(
            srt.Subtitle(
                index=idx,
                start=start_td,
                end=end_td,
                content=text,
            )
        )
    return srt.compose(subtitles)


def export_captions_json(
    segments: List[Segment],
    summary: str,
    metadata: VideoMetadata,
    output_path: str | Path,
) -> None:
    """Export complete captioning metadata to JSON."""
    data = {
        "video_path": metadata.path,
        "duration_sec": metadata.duration_sec,
        "fps": metadata.fps,
        "width": metadata.width,
        "height": metadata.height,
        "has_audio": metadata.has_audio,
        "summary": summary,
        "segments": [
            {
                "index": idx,
                "start_sec": seg.start_sec,
                "end_sec": seg.end_sec,
                "duration_sec": seg.duration_sec,
                "caption": seg.final_caption or seg.most_frequent_caption,
                "raw_blip_captions": seg.frame_captions,
                "timestamps": seg.timestamps,
            }
            for idx, seg in enumerate(segments)
        ],
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved captions JSON to {out_file}")


def escape_ffmpeg_path(path_str: str) -> str:
    """Escape windows file paths for ffmpeg filter syntax."""
    # Convert backslashes to forward slashes and escape colon
    clean = path_str.replace("\\", "/")
    clean = clean.replace(":", "\\:")
    clean = clean.replace("'", "\\'")
    clean = clean.replace("[", "\\[").replace("]", "\\]")
    return clean


def has_arabic_text(text: str) -> bool:
    """Detect if text contains Arabic characters."""
    return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", text))


def reshape_arabic_if_needed(text: str) -> str:
    """Reshape Arabic text for proper RTL display if libraries are installed."""
    if not has_arabic_text(text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)
    except ImportError:
        logger.warning("arabic-reshaper and python-bidi not installed. Arabic text might not render RTL.")
        return text


def burn_in_subtitles_ffmpeg(
    video_path: str,
    srt_path: str,
    output_path: str,
    metadata: VideoMetadata,
    config: Config,
) -> bool:
    """Burn in subtitles using ffmpeg subtitles filter."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    escaped_srt = escape_ffmpeg_path(srt_path)

    font_size = config.caption_font_size
    alignment = "2" if config.caption_position == "bottom" else "6"  # 2: bottom center, 6: top center
    margin_v = "24"

    # Style: white text, black outline, semi-transparent bounding box
    force_style = (
        f"FontSize={font_size},"
        f"Alignment={alignment},"
        f"MarginV={margin_v},"
        f"Outline=2,"
        f"Shadow=0,"
        f"BorderStyle=1,"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000"
    )

    filter_str = f"subtitles='{escaped_srt}':force_style='{force_style}',scale=trunc(iw/2)*2:trunc(ih/2)*2"

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        video_path,
        "-vf",
        filter_str,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-profile:v",
        "main",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]

    if metadata.has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k", "-ar", "44100"])
    else:
        cmd.append("-an")

    cmd.append(output_path)

    try:
        logger.info(f"Running ffmpeg subtitle burn-in...")
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if proc.returncode == 0 and Path(output_path).is_file():
            logger.info(f"Subtitles burned successfully: {output_path}")
            return True
        else:
            logger.warning(f"ffmpeg burn-in failed (rc={proc.returncode}): {proc.stderr}")
            return False
    except Exception as e:
        logger.warning(f"ffmpeg burn-in exception: {e}")
        return False


def mux_soft_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    metadata: VideoMetadata,
) -> bool:
    """Create video with embedded soft (toggleable) subtitle track."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        video_path,
        "-i",
        srt_path,
        "-c:v",
        "copy",
        "-c:s",
        "mov_text",
        "-metadata:s:s:0",
        "language=eng",
        "-movflags",
        "+faststart",
    ]

    if metadata.has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    else:
        cmd.append("-an")

    cmd.append(output_path)

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        return proc.returncode == 0 and Path(output_path).is_file()
    except Exception as e:
        logger.warning(f"Failed to mux soft subtitles: {e}")
        return False


def burn_in_subtitles_pillow_fallback(
    video_path: str,
    segments: List[Segment],
    output_path: str,
    metadata: VideoMetadata,
    config: Config,
) -> bool:
    """Fallback renderer: draw subtitles frame-by-frame with Pillow and mux audio."""
    logger.info("Using Pillow + OpenCV fallback subtitle renderer...")
    temp_dir = tempfile.mkdtemp(prefix="vidcap_render_")
    silent_temp_video = str(Path(temp_dir) / "silent_rendered.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = metadata.fps if metadata.fps > 0 else 25.0
    width = metadata.width
    height = metadata.height

    writer = cv2.VideoWriter(silent_temp_video, fourcc, fps, (width, height))

    # Font setup
    font_size = max(16, int(config.caption_font_size * (height / 720.0)))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Pre-build timeline lookup
    def get_caption_for_time(t_sec: float) -> str:
        for seg in segments:
            if seg.start_sec <= t_sec <= seg.end_sec:
                return seg.final_caption or seg.most_frequent_caption
        return ""

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_t = frame_idx / fps
            caption = get_caption_for_time(current_t)

            if caption:
                caption = reshape_arabic_if_needed(caption)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img)

                # Word wrap
                words = caption.split()
                lines = []
                curr_line = []
                max_w = int(width * 0.85)

                for w in words:
                    test_line = " ".join(curr_line + [w])
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    line_w = bbox[2] - bbox[0]
                    if line_w <= max_w:
                        curr_line.append(w)
                    else:
                        if curr_line:
                            lines.append(" ".join(curr_line))
                        curr_line = [w]
                if curr_line:
                    lines.append(" ".join(curr_line))

                # Compute bounding box
                line_heights = [draw.textbbox((0, 0), l, font=font)[3] - draw.textbbox((0, 0), l, font=font)[1] for l in lines]
                total_text_h = sum(line_heights) + (len(lines) - 1) * 8

                if config.caption_position == "top":
                    base_y = int(height * 0.08)
                else:
                    base_y = int(height * 0.88 - total_text_h)

                y = base_y
                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    line_w = bbox[2] - bbox[0]
                    x = (width - line_w) // 2
                    # Semi-transparent rectangle
                    pad_x, pad_y = 12, 6
                    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    draw_ol = ImageDraw.Draw(overlay)
                    draw_ol.rectangle(
                        [x - pad_x, y - pad_y, x + line_w + pad_x, y + line_heights[0] + pad_y],
                        fill=(0, 0, 0, 160),
                    )
                    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                    draw = ImageDraw.Draw(img)

                    # Text with black outline
                    for ox in (-1, 0, 1):
                        for oy in (-1, 0, 1):
                            draw.text((x + ox, y + oy), line, font=font, fill=(0, 0, 0))
                    draw.text((x, y), line, font=font, fill=(255, 255, 255))
                    y += line_heights[0] + 8

                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    # Mux audio back
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe, "-y", "-i", silent_temp_video]
    if metadata.has_audio:
        cmd.extend([
            "-i", video_path,
            "-c:v", "libx264",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-c:a", "aac",
            "-b:a", "128k",
            "-map", "0:v:0",
            "-map", "1:a:0?",
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
        ])

    cmd.append(output_path)

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception as e:
        logger.error(f"Fallback audio muxing failed: {e}")
        shutil.copyfile(silent_temp_video, output_path)
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def render_all_outputs(
    video_path: str | Path,
    segments: List[Segment],
    summary: str,
    metadata: VideoMetadata,
    config: Optional[Config] = None,
) -> tuple[str, str, str, Optional[str]]:
    """
    Render all project deliverables:
    - Burned-in MP4 (or original if burn-in=False)
    - SRT subtitle file
    - JSON metadata file
    - Soft-subtitles MP4 (if enabled)
    Returns: (output_video_path, srt_path, json_path, soft_subs_path)
    """
    cfg = config or Config()
    base_name = Path(video_path).stem
    video_out_dir = Path(cfg.output_dir) / base_name
    video_out_dir.mkdir(parents=True, exist_ok=True)

    srt_path = str(video_out_dir / f"{base_name}_captions.srt")
    json_path = str(video_out_dir / f"{base_name}_captions.json")
    burned_path = str(video_out_dir / f"{base_name}_captioned.mp4")
    soft_path = str(video_out_dir / f"{base_name}_softsubs.mp4") if cfg.also_soft_subs else None

    # 1. Write SRT
    srt_content = segments_to_srt(segments)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    logger.info(f"Saved SRT file to {srt_path}")

    # 2. Write JSON
    export_captions_json(segments, summary, metadata, json_path)

    # 3. Burn-in subtitles
    final_video_path = str(video_path)
    if cfg.burn_in:
        success = burn_in_subtitles_ffmpeg(
            video_path=str(video_path),
            srt_path=srt_path,
            output_path=burned_path,
            metadata=metadata,
            config=cfg,
        )
        if not success:
            logger.warning("ffmpeg burn-in failed. Trying Pillow fallback...")
            success = burn_in_subtitles_pillow_fallback(
                video_path=str(video_path),
                segments=segments,
                output_path=burned_path,
                metadata=metadata,
                config=cfg,
            )
        if success:
            final_video_path = burned_path

    # 4. Soft subtitles
    if soft_path and cfg.also_soft_subs:
        mux_soft_subtitles(
            video_path=str(video_path),
            srt_path=srt_path,
            output_path=soft_path,
            metadata=metadata,
        )

    return final_video_path, srt_path, json_path, soft_path
