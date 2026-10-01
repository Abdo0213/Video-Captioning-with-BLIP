import argparse
import logging
from pathlib import Path
import sys

# Ensure src/ is in python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from tqdm import tqdm

from vidcap.config import load_config
from vidcap.pipeline import CaptionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batched BLIP + Ollama Cloud Video Captioning System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video_path", type=str, help="Path to input video file (e.g. samples/test.mp4)")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "uniform", "scene"],
        default=None,
        help="Frame sampling mode",
    )
    parser.add_argument("--sample-every", type=float, default=None, help="Sample step in seconds")
    parser.add_argument("--batch-size", type=int, default=None, help="BLIP batch size")
    parser.add_argument("--blip-model", type=str, default=None, help="Hugging Face BLIP model name")
    parser.add_argument("--blip-device", type=str, default=None, help="Device (auto, cpu, cuda, mps)")
    parser.add_argument("--text-model", type=str, default=None, help="Ollama LLM model for text merge")
    parser.add_argument("--vision-model", type=str, default=None, help="Ollama Vision model for refine")
    parser.add_argument("--vision-refine", action="store_true", default=None, help="Enable vision refinement")
    parser.add_argument("--lang", type=str, default=None, help="Target language (en, ar, etc.)")
    parser.add_argument("--burn-in", action="store_true", default=None, help="Burn subtitles into output video")
    parser.add_argument("--no-burn-in", action="store_false", dest="burn_in", help="Do not burn subtitles")
    parser.add_argument("--soft-subs", action="store_true", default=None, help="Mux soft subtitle track")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for output files")
    parser.add_argument("--env", type=str, default=None, help="Path to .env configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def main():
    args = parse_args()

    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    overrides = {}
    if args.mode:
        overrides["sampling_mode"] = args.mode
    if args.sample_every is not None:
        overrides["sample_every_sec"] = args.sample_every
    if args.batch_size is not None:
        overrides["blip_batch_size"] = args.batch_size
    if args.blip_model:
        overrides["blip_model"] = args.blip_model
    if args.blip_device:
        overrides["blip_device"] = args.blip_device
    if args.text_model:
        overrides["ollama_text_model"] = args.text_model
    if args.vision_model:
        overrides["ollama_vision_model"] = args.vision_model
    if args.vision_refine is not None:
        overrides["refine_with_vision"] = args.vision_refine
    if args.lang:
        overrides["caption_language"] = args.lang
    if args.burn_in is not None:
        overrides["burn_in"] = args.burn_in
    if args.soft_subs is not None:
        overrides["also_soft_subs"] = args.soft_subs
    if args.output_dir:
        overrides["output_dir"] = args.output_dir

    config = load_config(env_path=args.env, **overrides)

    pbar = tqdm(total=100, desc="Video Captioning Progress", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}% [{elapsed}<{remaining}] {postfix}")

    def progress_callback(pct: float, msg: str):
        target_n = int(pct * 100)
        pbar.n = min(100, target_n)
        pbar.set_postfix_str(msg[:45])
        pbar.refresh()

    try:
        pipeline = CaptionPipeline(config=config)
        result = pipeline.run(args.video_path, progress_callback=progress_callback)
        pbar.n = 100
        pbar.close()

        print("\n" + "=" * 60)
        print("  VIDEO CAPTIONING SUMMARY")
        print("=" * 60)
        print(f"Summary: {result.summary}\n")
        print(f"Video Output: {result.output_video_path}")
        print(f"SRT Subtitles: {result.srt_path}")
        print(f"JSON Metadata: {result.json_path}")
        if result.soft_subs_path:
            print(f"Soft Subtitles Video: {result.soft_subs_path}")
        print("\nSegment Timeline:")
        for seg in result.segments:
            cap = seg.final_caption or seg.most_frequent_caption
            print(f"  [{seg.start_sec:5.2f}s -> {seg.end_sec:5.2f}s] {cap}")
        print("=" * 60)
    except Exception as e:
        pbar.close()
        logging.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
