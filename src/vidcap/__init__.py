"""Video Captioning package."""

from vidcap.config import Config, load_config
from vidcap.pipeline import CaptionPipeline, PipelineResult

__all__ = ["Config", "load_config", "CaptionPipeline", "PipelineResult"]
