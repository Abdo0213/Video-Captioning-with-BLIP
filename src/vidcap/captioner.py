"""Batched BLIP captioning and caption cleaning module."""

from __future__ import annotations

import logging
import re
from typing import Callable, Iterable, List, Optional
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from vidcap.config import Config
from vidcap.sampler import FrameSample

logger = logging.getLogger(__name__)


def clean_blip_caption(caption: str) -> str:
    """
    Clean raw BLIP caption:
    - Remove known BLIP artifact word 'arafed'
    - Remove consecutive duplicate words / repeated phrases
    - Fix punctuation and spacing
    """
    if not caption or not isinstance(caption, str):
        return ""

    text = caption.strip()

    # 1. Remove 'arafed' artifact (case-insensitive, whole word)
    text = re.sub(r"\barafed\b", "", text, flags=re.IGNORECASE)

    # 2. Fix multiple spaces / tabs
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Deduplicate consecutive identical words (e.g., "the the" -> "the")
    text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)

    # 4. Remove duplicate phrases (e.g., "a man walking a man walking" -> "a man walking")
    # Using regex to catch identical repeated segments
    for _ in range(2):
        text = re.sub(r"\b([a-zA-Z0-9\s]{3,25})\s+\1\b", r"\1", text, flags=re.IGNORECASE)

    # 5. Clean punctuation glitches (e.g. ", .", leading commas)
    text = re.sub(r"^[,.:;\s]+", "", text)
    text = re.sub(r"[,.:;\s]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    # Capitalize first character
    return text[0].upper() + text[1:] if len(text) > 1 else text.upper()


class BLIPCaptioner:
    """Wrapper around Hugging Face BLIP model with batched inference & OOM resilience."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.device = self._resolve_device(self.config.blip_device)
        self.dtype = (
            torch.float16
            if (self.device.type == "cuda" and self.config.blip_fp16)
            else torch.float32
        )

        logger.info(f"Loading BLIP model: {self.config.blip_model} on {self.device} ({self.dtype})")
        self.processor = BlipProcessor.from_pretrained(self.config.blip_model)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.config.blip_model,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        logger.info("BLIP model loaded successfully.")

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_str)

    def caption_images(
        self,
        images: List[Image.Image],
        batch_size: Optional[int] = None,
        prompt: Optional[str] = None,
    ) -> List[str]:
        """
        Run batched captioning on a list of PIL Images.
        Guarantees batch processing with single generate() call per batch.
        """
        if not images:
            return []

        bs = batch_size or self.config.blip_batch_size
        bs = max(1, bs)
        prompt_text = prompt if prompt is not None else self.config.blip_prompt

        all_captions: List[str] = []

        i = 0
        while i < len(images):
            batch_slice = images[i : i + bs]
            try:
                with torch.inference_mode():
                    if prompt_text and prompt_text.strip():
                        prompts = [prompt_text.strip()] * len(batch_slice)
                        inputs = self.processor(
                            images=batch_slice,
                            text=prompts,
                            return_tensors="pt",
                        ).to(self.device, self.dtype)
                    else:
                        inputs = self.processor(
                            images=batch_slice,
                            return_tensors="pt",
                        ).to(self.device, self.dtype)

                    outputs = self.model.generate(
                        **inputs,
                        num_beams=self.config.blip_num_beams,
                        max_new_tokens=self.config.blip_max_new_tokens,
                    )
                    raw_captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
                    cleaned = [clean_blip_caption(c) for c in raw_captions]
                    all_captions.extend(cleaned)
                i += len(batch_slice)
            except torch.cuda.OutOfMemoryError:
                if self.device.type == "cuda" and bs > 1:
                    logger.warning(
                        f"CUDA OutOfMemory with batch_size={bs}. Halving batch size to {bs // 2} and retrying."
                    )
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                else:
                    raise
            except Exception as e:
                if "CUDA out of memory" in str(e) and bs > 1:
                    logger.warning(
                        f"CUDA OOM with batch_size={bs}. Halving batch size to {bs // 2} and retrying."
                    )
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                else:
                    raise

        return all_captions

    def caption_samples(
        self,
        samples: List[FrameSample],
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[tuple[FrameSample, str]]:
        """Caption FrameSamples and return list of (sample, cleaned_caption)."""
        if not samples:
            return []

        bs = batch_size or self.config.blip_batch_size
        results: List[tuple[FrameSample, str]] = []

        total = len(samples)
        for start_idx in range(0, total, bs):
            chunk = samples[start_idx : start_idx + bs]
            images = [s.image for s in chunk]
            captions = self.caption_images(images, batch_size=len(images))
            for s, cap in zip(chunk, captions):
                results.append((s, cap))
            if progress_callback:
                progress_callback(min(start_idx + bs, total), total)

        return results
