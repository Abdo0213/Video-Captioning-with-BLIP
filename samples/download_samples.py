"""Download all sample test videos for video captioning demo & benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import urllib.request
import imageio_ffmpeg
from tqdm import tqdm


# Video name -> list of candidate URLs to try in order
SAMPLE_VIDEOS = {
    "test.mp4": [
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
    ],
    "bbb_short.mp4": [
        "https://raw.githubusercontent.com/cseitz/sample-files/main/assets/video/mp4/bbb_short.mp4",
        "https://raw.githubusercontent.com/bower-media-samples/big-buck-bunny-480p-30s/master/video.mp4",
    ],
    "escapes.mp4": [
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4",
    ],
    "fun.mp4": [
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4",
    ],
    "joyrides.mp4": [
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/bottle-detection.mp4",
    ],
    "gti_full.mp4": [
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/head-pose-face-detection-female.mp4",
    ],
    "street.mp4": [
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/driver-action-recognition.mp4",
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4",
    ],
    # --- Car / Driving Videos ---
    "what_car.mp4": [
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/vehicle-detection.mp4",
    ],
    "bullrun.mp4": [
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
    ],
    "subaru.mp4": [
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/driver-action-recognition.mp4",
    ],
    # --- Short Films (Blender Open Movies / CGI) ---
    "tears_of_steel.mp4": [
        "https://archive.org/download/Tears-of-Steel/tears_of_steel_720p.mp4",
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4",
    ],
    "sintel.mp4": [
        "https://archive.org/download/Sintel/sintel-2048-surround.mp4",
        "https://raw.githubusercontent.com/cseitz/sample-files/main/assets/video/mp4/bbb_short.mp4",
    ],
    "elephants_dream.mp4": [
        "https://archive.org/download/ElephantsDream/ed_1024_512kb.mp4",
        "https://raw.githubusercontent.com/bower-media-samples/big-buck-bunny-480p-30s/master/video.mp4",
    ],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def download_single_video(filename: str, urls: list[str], target_path: Path) -> bool:
    if target_path.exists() and target_path.stat().st_size > 1024:
        print(f"  [EXISTS] {filename:<20} ({target_path.stat().st_size / (1024*1024):.2f} MB)")
        return True

    for url in urls:
        print(f"  [DOWNLOADING] {filename} from {url}...")
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=45) as response:
                total_size = int(response.headers.get("content-length", 0))
                with open(target_path, "wb") as f, tqdm(
                    desc=filename,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=True,
                ) as bar:
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))
            print(f"  [SUCCESS] {filename} ({target_path.stat().st_size / (1024*1024):.2f} MB)")
            return True
        except Exception as e:
            print(f"  [RETRY] Error on {url}: {e}")

    return False


def cut_clip(source_path: Path, output_path: Path, start_time: str = "00:00:02", duration: int = 15):
    if output_path.exists() and output_path.stat().st_size > 50 * 1024:
        print(f"  [EXISTS] {output_path.name:<20} ({output_path.stat().st_size / (1024*1024):.2f} MB)")
        return

    if not source_path.exists() or source_path.stat().st_size < 1024:
        return

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Probe for audio stream
    probe_proc = subprocess.run(
        [ffmpeg_exe, "-i", str(source_path)],
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    has_audio = "Audio:" in (probe_proc.stderr or "")

    cmd = [
        ffmpeg_exe,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        start_time,
        "-i",
        str(source_path),
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ]
    if has_audio:
        cmd.extend(["-c:a", "aac"])
    else:
        cmd.append("-an")

    cmd.append(str(output_path))

    try:
        print(f"  [CUTTING] {output_path.name} ({duration}s from {start_time}, audio={has_audio})...")
        subprocess.run(cmd, check=True)
        print(f"  [SUCCESS] Created {output_path.name} ({output_path.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"  [WARN] Could not cut clip: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download sample videos or cut clips.")
    parser.add_argument("name", nargs="?", default=None, help="Specific file to download (leave blank for all)")
    parser.add_argument("--cut", nargs=2, metavar=("SOURCE", "OUTPUT"), help="Cut a clip: --cut input.mp4 output.mp4")
    parser.add_argument("--start", default="00:00:02", help="Start timestamp (e.g. 00:00:10)")
    parser.add_argument("--duration", type=int, default=15, help="Clip duration in seconds")
    args = parser.parse_args()

    if args.cut:
        src = Path(args.cut[0])
        dst = Path(args.cut[1])
        cut_clip(src, dst, start_time=args.start, duration=args.duration)
        return

    out_dir = Path("samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  DOWNLOADING SAMPLE VIDEOS")
    print("=" * 65)

    if args.name and args.name in SAMPLE_VIDEOS:
        download_single_video(args.name, SAMPLE_VIDEOS[args.name], out_dir / args.name)
    else:
        for name, urls in SAMPLE_VIDEOS.items():
            download_single_video(name, urls, out_dir / name)

        # Auto-generate gti_30s.mp4 clip
        gti_full = out_dir / "gti_full.mp4"
        gti_30s = out_dir / "gti_30s.mp4"
        if gti_full.exists():
            cut_clip(gti_full, gti_30s, start_time="00:00:02", duration=15)

    print("\n" + "=" * 65)
    print("  DOWNLOAD COMPLETE - FILES IN ./samples:")
    print("=" * 65)
    for f in sorted(out_dir.glob("*.mp4")):
        print(f"  ✓ {f.name:<22} ({f.stat().st_size / (1024*1024):6.2f} MB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
