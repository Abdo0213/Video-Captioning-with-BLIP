"""Full end-to-end Video Captioning Pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable, List, Optional

from vidcap.captioner import BLIPCaptioner
from vidcap.config import Config, load_config
from vidcap.llm import OllamaCaptionMerger
from vidcap.probe import VideoMetadata, probe_video
from vidcap.renderer import render_all_outputs
from vidcap.sampler import FrameSample, sample_frames_generator
from vidcap.segmenter import Segment, segment_frames

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    video_path: str
    output_video_path: str
    srt_path: str
    json_path: str
    soft_subs_path: Optional[str]
    summary: str
    segments: List[Segment]
    metadata: VideoMetadata


ProgressCallback = Callable[[float, str], None]


class CaptionPipeline:
    """End-to-end video captioning pipeline combining BLIP and Ollama Cloud."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self._captioner: Optional[BLIPCaptioner] = None
        self._llm: Optional[OllamaCaptionMerger] = None

    @property
    def captioner(self) -> BLIPCaptioner:
        if self._captioner is None:
            self._captioner = BLIPCaptioner(self.config)
        return self._captioner

    @property
    def llm(self) -> OllamaCaptionMerger:
        if self._llm is None:
            self._llm = OllamaCaptionMerger(self.config)
            self._llm.validate_models()
        return self._llm

    def run(
        self,
        video_path: str | Path,
        progress_callback: Optional[ProgressCallback] = None,
        **override_kwargs,
    ) -> PipelineResult:
        """Run complete video captioning pipeline."""
        path_str = str(Path(video_path).resolve())
        if not Path(path_str).is_file():
            raise FileNotFoundError(f"Input video not found: {path_str}")

        cfg = self.config
        if override_kwargs:
            cfg = load_config(**override_kwargs)

        def report(pct: float, msg: str):
            logger.info(f"[{int(pct * 100)}%] {msg}")
            if progress_callback:
                progress_callback(pct, msg)

        # Stage 1: Video Probing
        report(0.05, "Probing video metadata and audio tracks...")
        metadata = probe_video(path_str)

        # Stage 2: Streaming Frame Sampling & Batched BLIP Captioning
        report(0.15, "Sampling video frames (scene/hybrid)...")
        captioned_samples: List[tuple[FrameSample, str]] = []

        chunk_count = 0
        for frame_chunk in sample_frames_generator(path_str, config=cfg, metadata=metadata, chunk_size=32):
            chunk_count += 1
            report(
                0.20 + min(0.30, chunk_count * 0.05),
                f"Batched BLIP captioning (chunk {chunk_count}, {len(frame_chunk)} frames)...",
            )
            chunk_results = self.captioner.caption_samples(frame_chunk)
            captioned_samples.extend(chunk_results)

        if not captioned_samples:
            raise RuntimeError("No frames could be extracted or captioned from video.")

        logger.info(f"Extracted and captioned {len(captioned_samples)} total frames.")

        # Stage 3: Temporal Segmentation & Deduplication
        report(0.60, "Performing temporal segmentation and similarity clustering...")
        segments = segment_frames(captioned_samples, metadata.duration_sec, config=cfg)
        logger.info(f"Segmented video into {len(segments)} coherent segments.")

        # Stage 4: LLM Merge & Summarization via Ollama Cloud
        report(0.75, "Merging segment captions and summarizing with Ollama Cloud LLM...")
        merged_segments, summary = self.llm.merge_segments(segments, metadata.duration_sec)

        # Stage 5: Output Rendering (SRT, JSON, Video Burn-in)
        report(0.90, "Rendering output artifacts (SRT, JSON, burned-in video)...")
        out_video, srt_path, json_path, soft_path = render_all_outputs(
            video_path=path_str,
            segments=merged_segments,
            summary=summary,
            metadata=metadata,
            config=cfg,
        )

        report(1.0, "Video captioning complete!")

        return PipelineResult(
            video_path=path_str,
            output_video_path=out_video,
            srt_path=srt_path,
            json_path=json_path,
            soft_subs_path=soft_path,
            summary=summary,
            segments=merged_segments,
            metadata=metadata,
        )
