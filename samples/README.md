# Sample Test Videos

Place sample videos here for testing and benchmarking (e.g. `samples/test.mp4`).

### Quick clip extraction with bundled ffmpeg:
```bash
python -c "import imageio_ffmpeg, subprocess; subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-ss', '00:00:10', '-t', '30', '-i', 'your_video.mp4', '-c:v', 'libx264', '-c:a', 'aac', 'samples/test.mp4'])"
```
