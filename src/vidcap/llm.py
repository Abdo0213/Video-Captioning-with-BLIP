"""LLM merge and vision refinement module using Ollama Cloud."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import ollama

from vidcap.config import Config
from vidcap.segmenter import Segment

logger = logging.getLogger(__name__)


def image_to_base64_bytes(img: Image.Image, max_side: int = 384, quality: int = 80) -> bytes:
    """Convert PIL image to compressed JPEG bytes."""
    width, height = img.size
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / float(longest)
        img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR)

    if img.mode != "RGB":
        img = img.convert("RGB")

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


class OllamaCaptionMerger:
    """Handles LLM merge and vision refinement of video captions via Ollama Cloud."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.client = self._init_client()

    def _init_client(self) -> ollama.Client:
        headers = {}
        if self.config.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.config.ollama_api_key}"
        elif "ollama.com" in self.config.ollama_host.lower():
            logger.warning("OLLAMA_API_KEY is not set. Requests to Ollama Cloud may fail.")

        return ollama.Client(host=self.config.ollama_host, headers=headers)

    def validate_models(self) -> None:
        """Verify model availability and check vision capabilities if enabled."""
        if not self.config.ollama_api_key and "ollama.com" in self.config.ollama_host.lower():
            raise ValueError(
                "OLLAMA_API_KEY is required for Ollama Cloud. Please set it in your .env file."
            )

        try:
            available = self.client.list()
            model_names = [m.model for m in available.models] if hasattr(available, "models") else []
            logger.info(f"Available Ollama models: {model_names}")
        except Exception as e:
            logger.warning(f"Could not list models from Ollama host ({self.config.ollama_host}): {e}")

        if self.config.refine_with_vision:
            vision_model = self.config.ollama_vision_model
            try:
                info = self.client.show(vision_model)
                capabilities = getattr(info, "capabilities", None)
                details = getattr(info, "details", None)
                modalities = getattr(details, "modalities", []) if details else []

                is_vision = False
                if capabilities and "vision" in capabilities:
                    is_vision = True
                elif modalities and "vision" in modalities:
                    is_vision = True
                elif "vision" in vision_model.lower() or "gemma4" in vision_model.lower():
                    is_vision = True

                if not is_vision:
                    logger.warning(
                        f"Configured vision model '{vision_model}' might not report vision modality."
                    )
            except Exception as e:
                logger.warning(f"Could not verify capabilities for vision model '{vision_model}': {e}")

    def _call_with_retry(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = "json",
        images: Optional[List[bytes]] = None,
        max_retries: int = 3,
    ) -> str:
        """Call Ollama API with exponential backoff on rate limits / server errors."""
        delay = 1.0
        for attempt in range(max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "options": {"temperature": self.config.ollama_temperature},
                }
                if format:
                    kwargs["format"] = format

                response = self.client.chat(**kwargs)
                content = response.message.content if hasattr(response, "message") else response["message"]["content"]
                return content
            except Exception as e:
                err_str = str(e).lower()
                is_retryable = "429" in err_str or "500" in err_str or "502" in err_str or "503" in err_str or "timeout" in err_str
                if is_retryable and attempt < max_retries - 1:
                    logger.warning(f"Ollama request error: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    raise

        raise RuntimeError("Max retries exceeded for Ollama API call.")

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Safely parse JSON response with markdown code block cleaning."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except Exception:
            # Try to locate JSON dict substring
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
        return None

    def merge_segments(
        self,
        segments: List[Segment],
        video_duration: float,
    ) -> Tuple[List[Segment], str]:
        """
        Merge raw frame captions into coherent segment captions and a video summary.
        Falls back to most frequent raw captions on failure.
        """
        if not segments:
            return [], "No video segments found."

        # If Ollama is not configured or disabled, fallback directly
        if not self.config.ollama_api_key and "ollama.com" in self.config.ollama_host.lower():
            logger.warning("No OLLAMA_API_KEY configured. Falling back to raw BLIP captions.")
            for s in segments:
                s.final_caption = s.most_frequent_caption
            return segments, "Video overview: " + " ".join([s.most_frequent_caption for s in segments[:3]])

        # Prepare segment metadata for LLM prompt
        seg_data = []
        for idx, seg in enumerate(segments):
            seg_data.append({
                "segment_index": idx,
                "start": f"{seg.start_sec:.2f}s",
                "end": f"{seg.end_sec:.2f}s",
                "raw_blip_captions": seg.frame_captions,
            })

        system_prompt = (
            "You are an expert video captioning and summarization AI.\n"
            "Given a list of timestamped video segments with raw, noisy frame-level captions from an image captioner:\n"
            "1. Produce ONE concise, natural, fluent caption per segment in present tense (at most 15 words).\n"
            "2. Do not invent details beyond the visual evidence described in the raw captions.\n"
            f"3. Write the captions and summary in the requested language: '{self.config.caption_language}'.\n"
            "4. Provide a single, coherent one-sentence summary of the entire video.\n"
            "5. Respond STRICTLY with a valid JSON object in the following format:\n"
            "{\n"
            '  "video_summary": "A coherent single sentence summary of the whole video.",\n'
            '  "segments": [\n'
            '    {"segment_index": 0, "caption": "Natural fluent caption for segment 0"},\n'
            '    {"segment_index": 1, "caption": "Natural fluent caption for segment 1"}\n'
            "  ]\n"
            "}"
        )

        user_prompt = f"Video duration: {video_duration:.2f} seconds.\nSegments to merge:\n{json.dumps(seg_data, indent=2)}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        parsed: Optional[Dict[str, Any]] = None
        try:
            resp_text = self._call_with_retry(
                model=self.config.ollama_text_model,
                messages=messages,
                format="json",
            )
            parsed = self._parse_json_response(resp_text)

            # If invalid, attempt one repair step
            if not parsed or "segments" not in parsed:
                logger.warning("Malformed JSON from Ollama. Attempting repair prompt...")
                repair_messages = messages + [
                    {"role": "assistant", "content": resp_text},
                    {
                        "role": "user",
                        "content": "Your previous response was not valid JSON or lacked the 'segments' array. Please output ONLY the valid JSON object now.",
                    },
                ]
                repair_text = self._call_with_retry(
                    model=self.config.ollama_text_model,
                    messages=repair_messages,
                    format="json",
                )
                parsed = self._parse_json_response(repair_text)
        except Exception as e:
            logger.error(f"LLM merge failed with error: {e}. Utilizing graceful fallback.")

        # Apply results or fallback
        video_summary = "Video overview."
        if parsed and isinstance(parsed, dict):
            video_summary = str(parsed.get("video_summary", "")).strip() or "Video overview."
            caption_map = {}
            for item in parsed.get("segments", []):
                if isinstance(item, dict) and "segment_index" in item and "caption" in item:
                    caption_map[int(item["segment_index"])] = str(item["caption"]).strip()

            for idx, seg in enumerate(segments):
                if idx in caption_map and caption_map[idx]:
                    seg.final_caption = caption_map[idx]
                else:
                    seg.final_caption = seg.most_frequent_caption
        else:
            # Fallback to most frequent BLIP caption
            for seg in segments:
                seg.final_caption = seg.most_frequent_caption
            video_summary = "Video sequence: " + "; ".join([s.most_frequent_caption for s in segments[:4]])

        # Optional: refine with vision if enabled
        if self.config.refine_with_vision:
            self._refine_with_vision(segments)

        return segments, video_summary

    def _refine_with_vision(self, segments: List[Segment]) -> None:
        """Refine segment captions using the vision model and attached keyframes."""
        logger.info(f"Refining captions with Ollama vision model: {self.config.ollama_vision_model}")

        def refine_single_segment(seg: Segment) -> None:
            if not seg.keyframes:
                return

            images_bytes = [image_to_base64_bytes(img) for img in seg.keyframes[:3]]
            prompt = (
                f"Look at the attached keyframes from a video segment ({seg.start_sec:.1f}s - {seg.end_sec:.1f}s). "
                f"Initial caption: '{seg.final_caption}'. "
                f"Provide a more accurate, concise (under 15 words) caption in '{self.config.caption_language}' "
                f"describing what is happening. Respond only with the corrected sentence."
            )
            try:
                response = self.client.chat(
                    model=self.config.ollama_vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": images_bytes,
                        }
                    ],
                    options={"temperature": 0.1},
                )
                refined = response.message.content if hasattr(response, "message") else response["message"]["content"]
                refined = refined.strip().strip('"').strip("'")
                if refined:
                    seg.final_caption = refined
            except Exception as e:
                logger.warning(f"Vision refinement failed for segment {seg.start_sec}-{seg.end_sec}: {e}")

        workers = max(1, self.config.ollama_concurrency)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(refine_single_segment, segments))
