"""Configuration management for Video Captioning."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
from dotenv import load_dotenv


def _to_bool(val: str | bool | None, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class Config:
    # BLIP local model settings
    blip_model: str = "Salesforce/blip-image-captioning-large"
    blip_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    blip_fp16: bool = True
    blip_batch_size: int = 16
    blip_num_beams: int = 3
    blip_max_new_tokens: int = 30
    blip_prompt: str = ""

    # Frame sampling
    sampling_mode: Literal["hybrid", "uniform", "scene"] = "hybrid"
    sample_every_sec: float = 1.0
    scene_threshold: float = 27.0
    max_frames: int = 300
    frame_max_side: int = 512

    # Segmenting & Merging
    segment_min_sec: float = 2.0
    segment_max_sec: float = 8.0
    dedupe_similarity: float = 90.0
    caption_language: str = "en"

    # Ollama Cloud
    ollama_host: str = "https://ollama.com"
    ollama_api_key: str = ""
    ollama_text_model: str = "gpt-oss:120b"
    ollama_vision_model: str = "gemma4:31b"
    refine_with_vision: bool = False
    ollama_temperature: float = 0.2
    ollama_concurrency: int = 4

    # Output & Rendering
    output_dir: str = "./outputs"
    caption_font_size: int = 28
    caption_position: Literal["bottom", "top"] = "bottom"
    burn_in: bool = True
    also_soft_subs: bool = True


def load_config(env_path: Optional[str | Path] = None, **overrides) -> Config:
    """Load configuration from environment variables / .env file with optional overrides."""
    if env_path is not None:
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=False)

    config = Config(
        # BLIP
        blip_model=os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-large"),
        blip_device=os.getenv("BLIP_DEVICE", "auto"),  # type: ignore
        blip_fp16=_to_bool(os.getenv("BLIP_FP16", "true"), default=True),
        blip_batch_size=int(os.getenv("BLIP_BATCH_SIZE", "16")),
        blip_num_beams=int(os.getenv("BLIP_NUM_BEAMS", "3")),
        blip_max_new_tokens=int(os.getenv("BLIP_MAX_NEW_TOKENS", "30")),
        blip_prompt=os.getenv("BLIP_PROMPT", ""),
        # Sampling
        sampling_mode=os.getenv("SAMPLING_MODE", "hybrid"),  # type: ignore
        sample_every_sec=float(os.getenv("SAMPLE_EVERY_SEC", "1.0")),
        scene_threshold=float(os.getenv("SCENE_THRESHOLD", "27.0")),
        max_frames=int(os.getenv("MAX_FRAMES", "300")),
        frame_max_side=int(os.getenv("FRAME_MAX_SIDE", "512")),
        # Segmenting
        segment_min_sec=float(os.getenv("SEGMENT_MIN_SEC", "2.0")),
        segment_max_sec=float(os.getenv("SEGMENT_MAX_SEC", "8.0")),
        dedupe_similarity=float(os.getenv("DEDUPE_SIMILARITY", "90")),
        caption_language=os.getenv("CAPTION_LANGUAGE", "en"),
        # Ollama
        ollama_host=os.getenv("OLLAMA_HOST", "https://ollama.com"),
        ollama_api_key=os.getenv("OLLAMA_API_KEY", ""),
        ollama_text_model=os.getenv("OLLAMA_TEXT_MODEL", "gpt-oss:120b"),
        ollama_vision_model=os.getenv("OLLAMA_VISION_MODEL", "gemma4:31b"),
        refine_with_vision=_to_bool(os.getenv("REFINE_WITH_VISION", "false"), default=False),
        ollama_temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
        ollama_concurrency=int(os.getenv("OLLAMA_CONCURRENCY", "4")),
        # Output
        output_dir=os.getenv("OUTPUT_DIR", "./outputs"),
        caption_font_size=int(os.getenv("CAPTION_FONT_SIZE", "28")),
        caption_position=os.getenv("CAPTION_POSITION", "bottom"),  # type: ignore
        burn_in=_to_bool(os.getenv("BURN_IN", "true"), default=True),
        also_soft_subs=_to_bool(os.getenv("ALSO_SOFT_SUBS", "true"), default=True),
    )

    for k, v in overrides.items():
        if v is not None and hasattr(config, k):
            setattr(config, k, v)

    return config
