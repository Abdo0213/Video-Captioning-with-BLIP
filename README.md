# 🎬 Video Captioning with BLIP (Batched) + Ollama Cloud

An end-to-end intelligent video captioning and subtitling system. It combines local **batched BLIP** vision models with **Ollama Cloud LLMs** to produce coherent, temporal segment captions, video summaries, and burned-in subtitled videos while preserving original audio.

---

## 🌟 Key Features

- **Batched BLIP Inference**: Frames are processed strictly in configurable batches (never one-by-one) with automatic CUDA OOM resilience and `arafed` artifact cleaning.
- **Intelligent Temporal Merge**: Uses Ollama Cloud LLMs (`gpt-oss:120b`, `glm-5.3-flash`, etc.) to turn noisy per-frame captions into fluent, coherent 15-word present-tense segment captions and a video summary.
- **Vision Refinement (Optional)**: Refines ambiguous segment captions using multimodal vision models (`gemma4:31b`) with attached keyframes.
- **Smart Frame Sampling**: Supports `hybrid` (PySceneDetect scene cuts + max time-gap fill), `scene`, and `uniform` modes with memory-bounded chunk streaming.
- **High-Quality Subtitle Rendering**: Burns subtitles into videos using ffmpeg (`subtitles` filter with `force_style`), preserves original audio streams, generates `.srt` and `.json` files, and embeds soft subtitle tracks (`mov_text`).
- **Pillow & Arabic Fallback**: Seamless Pillow-based rendering with RTL text support (`arabic-reshaper`, `python-bidi`) if ffmpeg `libass` is unavailable.
- **Interactive Gradio Web App & CLI**: User-friendly web UI with stage-by-stage progress tracking, video player, segment timeline table, and artifact download buttons.

---

## 🏗️ Architecture & Pipeline

```
Video Input ──► Probe (fps, duration, resolution, audio detection)
            ──► Streamed Frame Sampler (Hybrid / Scene / Uniform)
            ──► Batched BLIP Captioner (Hugging Face local inference)
            ──► Caption Cleaning ("arafed" & repetition stripping)
            ──► Temporal Segmenter (Deduplication & rapidfuzz clustering)
            ──► Ollama Cloud Merge (Structured JSON + Validation + Repair)
            ──► Subtitle & Video Renderer (FFmpeg burn-in + SRT + JSON + Soft-subs)
            ──► Captioned Output Video (Audio Preserved)
```

---

## 📁 Project Structure

```
video-captioning/
├── .env.example                # Template for environment variables and model configs
├── .env                        # Local configuration & Ollama API key
├── requirements.txt            # Project dependencies
├── README.md                   # Full documentation and guide
├── app.py                      # Modern Gradio web UI
├── cli.py                      # Command-line interface with progress bars
├── src/vidcap/
│   ├── __init__.py
│   ├── config.py               # Typed configuration loader (.env)
│   ├── probe.py                # Video & audio metadata extraction
│   ├── sampler.py              # Hybrid, scene, and uniform frame sampling
│   ├── captioner.py            # Batched BLIP captioning & artifact cleaning
│   ├── segmenter.py            # Temporal segmentation & similarity clustering
│   ├── llm.py                  # Ollama Cloud LLM merge, JSON repair & vision refine
│   ├── renderer.py             # Subtitle burn-in, soft-subs, and Pillow fallback
│   └── pipeline.py             # Complete end-to-end orchestration pipeline
├── samples/
│   ├── README.md               # Guidelines for test videos
│   ├── download_samples.py     # Script to download sample demo videos
│   └── test.mp4                # Downloaded sample video
├── eval/
│   ├── evaluate.py             # Speed benchmarking & CLIP score evaluation
│   └── results.json            # Benchmark output metrics
└── tests/
    ├── test_captioner.py       # Unit tests for BLIP batching & cleaning
    ├── test_segmenter.py       # Unit tests for segmentation & constraints
    ├── test_llm.py             # Unit tests for LLM JSON parse & fallback
    └── test_renderer.py        # Unit tests for SRT generation & ffmpeg paths
```

---

## 🚀 Setup & Installation

### 1. Activate Virtual Environment & Install Dependencies

```bash
# Windows
.venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and configure your settings:

```bash
copy .env.example .env
```

Edit `.env` to include your **Ollama Cloud API Key**:
```env
OLLAMA_HOST=https://ollama.com
OLLAMA_API_KEY=your_ollama_api_key_here
OLLAMA_TEXT_MODEL=gpt-oss:120b
OLLAMA_VISION_MODEL=gemma4:31b
```

---

## 🎬 How to Run

### 1. Download Sample Videos
Download ready-to-use sample test videos into the `samples/` folder:

```bash
python samples/download_samples.py test.mp4
# Or download animation / action clips:
python samples/download_samples.py bunny.mp4
python samples/download_samples.py street.mp4
```

### 2. Launch the Web UI (Gradio)

Start the interactive web application:

```bash
python app.py
```
Open `http://localhost:7860` in your browser. Upload your video, configure options in the settings accordion, and click **🚀 Generate Video Captions**.

### 3. Run via Command-Line Interface (CLI)

Process any video directly from the terminal:

```bash
# Basic run with hybrid sampling and subtitle burn-in
python cli.py samples/test.mp4 --mode hybrid --burn-in

# Custom options: change batch size, language, and output directory
python cli.py samples/test.mp4 --batch-size 8 --lang en --output-dir ./my_outputs --burn-in --soft-subs

# Run with vision refinement (using Gemma4 31B)
python cli.py samples/test.mp4 --vision-refine
```

### 4. Run Speed & Quality Evaluation

Benchmark BLIP throughput across different batch sizes (1, 4, 8, 16, 32) and evaluate caption quality:

```bash
python eval/evaluate.py --video samples/test.mp4 --output eval/results.json
```

### 5. Run the Test Suite

Verify all pipeline modules and acceptance criteria:

```bash
pytest
```

---

## ⚙️ Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `BLIP_MODEL` | `Salesforce/blip-image-captioning-large` | Hugging Face BLIP model identifier |
| `BLIP_DEVICE` | `auto` | Device to run BLIP (`auto`, `cuda`, `cpu`, `mps`) |
| `BLIP_FP16` | `true` | Use half-precision FP16 on CUDA |
| `BLIP_BATCH_SIZE` | `16` | Number of frames per `model.generate()` call |
| `SAMPLING_MODE` | `hybrid` | `hybrid` (scene + uniform gap fill), `scene`, or `uniform` |
| `SAMPLE_EVERY_SEC` | `1.0` | Maximum gap between sampled frames in seconds |
| `SCENE_THRESHOLD` | `27.0` | PySceneDetect ContentDetector sensitivity threshold |
| `MAX_FRAMES` | `300` | Hard cap on total sampled frames per video |
| `SEGMENT_MIN_SEC` | `2.0` | Minimum allowed duration for a subtitle segment |
| `SEGMENT_MAX_SEC` | `8.0` | Maximum allowed duration before splitting a segment |
| `DEDUPE_SIMILARITY` | `90` | Rapidfuzz ratio threshold for merging adjacent captions |
| `CAPTION_LANGUAGE` | `en` | Target language for final captions (`en`, `ar`, etc.) |
| `OLLAMA_HOST` | `https://ollama.com` | Ollama API endpoint |
| `OLLAMA_API_KEY` | `""` | Ollama Cloud authorization token |
| `OLLAMA_TEXT_MODEL` | `gpt-oss:120b` | LLM model for caption merging and summarization |
| `OLLAMA_VISION_MODEL`| `gemma4:31b` | Vision model for optional keyframe refinement |
| `REFINE_WITH_VISION` | `false` | Attach keyframes to vision model for caption correction |
| `BURN_IN` | `true` | Hard-burn subtitles directly onto output video |
| `ALSO_SOFT_SUBS` | `true` | Export secondary video with toggleable soft subtitle track |
| `OUTPUT_DIR` | `./outputs` | Directory where rendered files are saved |

---

## 💡 Known Behaviors & Fallbacks

1. **LLM Fault Tolerance**: If the Ollama Cloud service is unreachable or rate-limited, the pipeline never crashes; it automatically falls back to the most frequent raw BLIP caption for each segment.
2. **Audio Preservation**: Original audio tracks are losslessly copied into output MP4s.
3. **Low-Memory / CPU Support**: Set `BLIP_BATCH_SIZE=4` and `BLIP_MODEL=Salesforce/blip-image-captioning-base` in `.env` for lower RAM / CPU machines.
