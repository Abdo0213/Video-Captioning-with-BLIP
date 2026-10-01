import asyncio
import logging
import os
from pathlib import Path
import subprocess
import sys
import imageio_ffmpeg

# Ensure src/ is in python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def silence_asyncio_connection_reset(loop, context):
    """Filter out harmless Windows Proactor TCP reset errors when browser refreshes."""
    exception = context.get("exception")
    msg = str(context.get("message", ""))
    if isinstance(exception, ConnectionResetError) or "10054" in str(exception) or "10054" in msg:
        return
    loop.default_exception_handler(context)


# Set global exception handler for asyncio
try:
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(silence_asyncio_connection_reset)
except Exception:
    pass

import gradio as gr
import pandas as pd

from vidcap.config import Config, load_config
from vidcap.pipeline import CaptionPipeline
from vidcap.probe import probe_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Available Ollama Cloud models on this account
TEXT_MODELS = [
    "gpt-oss:120b",
    "glm-5.3-flash",
    "deepseek-v4.1-flash",
    "mistral-large-3:675b",
    "nemotron-3-super",
    "minimax-m2.7",
    "nemotron-3-nano:30b",
    "gpt-oss:20b",
    "nemotron-3-ultra",
    "glm-5.2",
    "kimi-k3",
    "kimi-k2.7-code",
    "glm-5.3",
    "kimi-k2.6",
    "minimax-m3",
    "deepseek-v4-pro:0813",
]
VISION_MODELS = ["gemma4:31b"]

CUSTOM_CSS = """
:root {
    --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    --card-bg: rgba(30, 41, 59, 0.7);
    --border-color: rgba(255, 255, 255, 0.1);
}
.header-box {
    background: var(--primary-gradient);
    padding: 24px;
    border-radius: 16px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.3);
}
.header-box h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 800;
}
.header-box p {
    margin: 6px 0 0 0;
    opacity: 0.9;
    font-size: 15px;
}
.summary-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    padding: 16px 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.summary-title {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #a5b4fc;
    font-weight: 700;
    margin-bottom: 6px;
}
.summary-text {
    font-size: 16px;
    line-height: 1.5;
    color: #f8fafc;
}
"""


def cut_video_clip_helper(input_path: str, start_sec: float, duration_sec: float) -> str:
    """Trim video cleanly using bundled FFmpeg with audio support."""
    out_dir = Path("outputs") / "trimmed_clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    out_path = str(out_dir / f"{stem}_cut_{int(start_sec)}s_{int(duration_sec)}s.mp4")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Probe audio
    probe_proc = subprocess.run(
        [ffmpeg_exe, "-i", input_path],
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    has_audio = "Audio:" in (probe_proc.stderr or "")

    cmd = [
        ffmpeg_exe,
        "-y",
        "-ss",
        f"{start_sec:.2f}",
        "-i",
        input_path,
        "-t",
        f"{duration_sec:.2f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    else:
        cmd.append("-an")

    cmd.append(out_path)
    subprocess.run(cmd, check=True)
    return out_path


def process_video(
    video_path: str,
    enable_trim: bool,
    trim_start: float,
    trim_duration: float,
    sampling_mode: str,
    sample_every_sec: float,
    scene_threshold: float,
    segment_min_sec: float,
    segment_max_sec: float,
    blip_batch_size: int,
    text_model: str,
    vision_model: str,
    refine_with_vision: bool,
    caption_lang: str,
    burn_in: bool,
    also_soft_subs: bool,
    progress=gr.Progress(),
):
    if not video_path:
        raise gr.Error("Please upload a video file first.")

    target_video = video_path
    if enable_trim and trim_duration > 0:
        progress(0.02, desc="Trimming input video clip...")
        target_video = cut_video_clip_helper(video_path, trim_start, trim_duration)

    cfg = load_config(
        sampling_mode=sampling_mode,
        sample_every_sec=sample_every_sec,
        scene_threshold=scene_threshold,
        segment_min_sec=segment_min_sec,
        segment_max_sec=segment_max_sec,
        blip_batch_size=blip_batch_size,
        ollama_text_model=text_model,
        ollama_vision_model=vision_model,
        refine_with_vision=refine_with_vision,
        caption_language=caption_lang,
        burn_in=burn_in,
        also_soft_subs=also_soft_subs,
    )

    def gradio_progress_cb(pct: float, msg: str):
        progress(pct, desc=msg)

    pipeline = CaptionPipeline(config=cfg)
    result = pipeline.run(target_video, progress_callback=gradio_progress_cb)

    # Prepare segments dataframe
    table_rows = []
    for s in result.segments:
        table_rows.append({
            "Start (s)": f"{s.start_sec:.2f}",
            "End (s)": f"{s.end_sec:.2f}",
            "Duration (s)": f"{s.duration_sec:.2f}",
            "Final Caption": s.final_caption or s.most_frequent_caption,
            "Raw BLIP Captions": " | ".join(s.frame_captions),
        })

    df = pd.DataFrame(table_rows)

    summary_html = f"""
    <div class="summary-card">
        <div class="summary-title">🎬 Video Summary</div>
        <div class="summary-text">{result.summary}</div>
    </div>
    """

    out_video = str(Path(result.output_video_path).resolve()) if os.path.exists(result.output_video_path) else str(Path(target_video).resolve())
    srt_out = str(Path(result.srt_path).resolve()) if result.srt_path and os.path.exists(result.srt_path) else None
    json_out = str(Path(result.json_path).resolve()) if result.json_path and os.path.exists(result.json_path) else None
    soft_out = str(Path(result.soft_subs_path).resolve()) if result.soft_subs_path and os.path.exists(result.soft_subs_path) else None

    return (
        out_video,
        summary_html,
        df,
        srt_out,
        json_out,
        soft_out,
    )


def create_app() -> gr.Blocks:
    with gr.Blocks(title="AI Video Captioning") as demo:
        gr.HTML("""
        <div class="header-box">
            <h1>🎬 Video Captioning with BLIP & Ollama Cloud</h1>
            <p>Batched frame captioning + intelligent temporal LLM merge with subtitle burn-in.</p>
        </div>
        """)

        with gr.Tabs():
            with gr.Tab("🚀 Full Captioning Pipeline"):
                with gr.Row():
                    with gr.Column(scale=5):
                        video_input = gr.Video(label="Upload Video", sources=["upload"])

                        with gr.Accordion("✂️ Trim / Cut Video (Optional)", open=False):
                            enable_trim = gr.Checkbox(label="Enable Video Trimming (Process specific clip)", value=False)
                            with gr.Row():
                                trim_start = gr.Number(label="Start Time (seconds)", value=0.0, minimum=0.0)
                                trim_duration = gr.Number(label="Duration (seconds)", value=30.0, minimum=1.0)

                        with gr.Accordion("⚙️ Advanced Pipeline Settings", open=False):
                            with gr.Row():
                                sampling_mode = gr.Dropdown(
                                    choices=["hybrid", "scene", "uniform"],
                                    value="hybrid",
                                    label="Sampling Mode",
                                )
                                sample_every_sec = gr.Slider(
                                    minimum=0.2, maximum=5.0, value=1.0, step=0.1, label="Sample Every (sec)"
                                )

                            with gr.Row():
                                scene_threshold = gr.Slider(
                                    minimum=5.0, maximum=60.0, value=27.0, step=1.0, label="Scene Cut Sensitivity (Threshold)"
                                )
                                blip_batch_size = gr.Slider(
                                    minimum=1, maximum=64, value=16, step=1, label="BLIP Batch Size"
                                )

                            with gr.Row():
                                segment_min_sec = gr.Slider(minimum=0.5, maximum=5.0, value=2.0, step=0.5, label="Min Segment (sec)")
                                segment_max_sec = gr.Slider(minimum=2.0, maximum=20.0, value=8.0, step=1.0, label="Max Segment (sec)")

                            with gr.Row():
                                text_model = gr.Dropdown(
                                    choices=TEXT_MODELS,
                                    value="gpt-oss:120b",
                                    label="Ollama Text Model (Merge)",
                                )
                                vision_model = gr.Dropdown(
                                    choices=VISION_MODELS,
                                    value="gemma4:31b",
                                    label="Ollama Vision Model",
                                )

                            with gr.Row():
                                caption_lang = gr.Textbox(value="en", label="Caption Language (e.g. en, ar)")
                                refine_with_vision = gr.Checkbox(
                                    value=False, label="Refine with Keyframe Vision (Gemma4 31B)"
                                )

                            with gr.Row():
                                burn_in = gr.Checkbox(value=True, label="Burn-in Subtitles to Video")
                                also_soft_subs = gr.Checkbox(value=True, label="Export Soft-subtitles Video")

                        submit_btn = gr.Button("🚀 Generate Video Captions", variant="primary", size="lg")

                    with gr.Column(scale=6):
                        video_output = gr.Video(label="Captioned Video Result", interactive=False)
                        summary_output = gr.HTML(label="Summary")

                        with gr.Tab("Segment Timeline"):
                            segments_table = gr.Dataframe(
                                headers=["Start (s)", "End (s)", "Duration (s)", "Final Caption", "Raw BLIP Captions"],
                                datatype=["str", "str", "str", "str", "str"],
                                interactive=False,
                            )

                        with gr.Row():
                            srt_file = gr.File(label="Download Subtitles (.srt)")
                            json_file = gr.File(label="Download Metadata (.json)")
                            soft_file = gr.File(label="Download Soft-Sub Video (.mp4)")

                submit_btn.click(
                    fn=process_video,
                    inputs=[
                        video_input,
                        enable_trim,
                        trim_start,
                        trim_duration,
                        sampling_mode,
                        sample_every_sec,
                        scene_threshold,
                        segment_min_sec,
                        segment_max_sec,
                        blip_batch_size,
                        text_model,
                        vision_model,
                        refine_with_vision,
                        caption_lang,
                        burn_in,
                        also_soft_subs,
                    ],
                    outputs=[
                        video_output,
                        summary_output,
                        segments_table,
                        srt_file,
                        json_file,
                        soft_file,
                    ],
                )

            with gr.Tab("✂️ Standalone Video Trimmer"):
                gr.Markdown("### Cut & Extract Video Clips without Captioning")
                with gr.Row():
                    with gr.Column():
                        trim_in_vid = gr.Video(label="Input Video")
                        with gr.Row():
                            t_start = gr.Number(label="Start Time (seconds)", value=0.0, minimum=0.0)
                            t_dur = gr.Number(label="Duration (seconds)", value=15.0, minimum=1.0)
                        cut_btn = gr.Button("✂️ Cut Clip", variant="secondary")
                    with gr.Column():
                        trim_out_vid = gr.Video(label="Trimmed Video Clip", interactive=False)

                cut_btn.click(
                    fn=cut_video_clip_helper,
                    inputs=[trim_in_vid, t_start, t_dur],
                    outputs=[trim_out_vid],
                )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        allowed_paths=[
            str(Path("outputs").resolve()),
            str(Path("samples").resolve()),
            str(Path(".").resolve()),
        ],
    )
