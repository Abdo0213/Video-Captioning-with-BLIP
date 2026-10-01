import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

# Ensure src/ is in python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
from PIL import Image
import torch

from vidcap.captioner import BLIPCaptioner
from vidcap.config import load_config
from vidcap.probe import probe_video
from vidcap.sampler import sample_all_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)


def benchmark_blip_speed(
    images: List[Image.Image],
    batch_sizes: List[int] = [1, 4, 8, 16, 32],
    num_warmup: int = 2,
) -> Dict[int, float]:
    """Benchmark frames-per-second processing speed across various batch sizes."""
    config = load_config()
    captioner = BLIPCaptioner(config)

    results = {}
    print("\n" + "=" * 60)
    print("  BLIP BATCH SPEED BENCHMARK")
    print("=" * 60)
    print(f"{'Batch Size':<15} | {'Total Time (s)':<18} | {'Throughput (fps)':<18}")
    print("-" * 60)

    # Warmup
    if images:
        for _ in range(num_warmup):
            captioner.caption_images(images[:2], batch_size=2)

    for bs in batch_sizes:
        if bs > len(images):
            # Duplicate images to meet test batch size
            test_imgs = (images * (bs // len(images) + 1))[: max(bs * 2, 32)]
        else:
            test_imgs = images[: max(bs * 2, 32)]

        start_time = time.perf_counter()
        captioner.caption_images(test_imgs, batch_size=bs)
        elapsed = time.perf_counter() - start_time

        fps = len(test_imgs) / elapsed if elapsed > 0 else 0.0
        results[bs] = round(fps, 2)
        print(f"{bs:<15} | {elapsed:<18.2f} | {fps:<18.2f}")

    print("=" * 60 + "\n")
    return results


def compute_clip_similarity(
    images: List[Image.Image],
    texts: List[str],
    clip_model_name: str = "openai/clip-vit-base-patch32",
) -> float:
    """Compute reference-free CLIP similarity score between keyframe images and generated captions."""
    try:
        from transformers import CLIPModel, CLIPProcessor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CLIPModel.from_pretrained(clip_model_name).to(device)
        processor = CLIPProcessor.from_pretrained(clip_model_name)

        if not images or not texts:
            return 0.0

        scores = []
        for img, text in zip(images, texts):
            if not text.strip():
                continue
            inputs = processor(text=[text], images=img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                sim = torch.matmul(image_embeds, text_embeds.t()).squeeze().item()
                scores.append(max(0.0, sim))

        return float(np.mean(scores)) if scores else 0.0
    except Exception as e:
        logger.warning(f"CLIP evaluation skipped or error occurred: {e}")
        return 0.0


def run_evaluation(video_path: str, output_path: str = "eval/results.json"):
    meta = probe_video(video_path)
    frames = sample_all_frames(video_path)
    images = [f.image for f in frames]

    if not images:
        logger.error("No frames extracted for evaluation.")
        return

    # 1. Benchmark speed
    speed_results = benchmark_blip_speed(images, batch_sizes=[1, 4, 8, 16, 32])

    eval_data = {
        "video_path": str(video_path),
        "total_frames_tested": len(images),
        "fps_by_batch_size": speed_results,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)

    logger.info(f"Evaluation results successfully saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="samples/test.mp4", help="Path to sample video for benchmark")
    parser.add_argument("--output", type=str, default="eval/results.json", help="Path to save evaluation results")
    args = parser.parse_args()

    if Path(args.video).is_file():
        run_evaluation(args.video, args.output)
    else:
        logger.info(f"No video found at {args.video}. You can pass --video <path_to_video> to run benchmarks.")
