# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "faster-whisper",
#     "tqdm",
#     "nvidia-cublas-cu12",
#     "nvidia-cudnn-cu12",
# ]
# ///

import ctypes
import importlib.util
import os
import sys
from pathlib import Path

from faster_whisper import WhisperModel
from tqdm import tqdm

import time



def load_cuda_libs():
    """Preload CUDA 12 libraries installed via wheels into CTranslate2's process space."""
    for mod_name in ["nvidia.cublas", "nvidia.cudnn"]:
        try:
            spec = importlib.util.find_spec(mod_name)
            if spec and spec.submodule_search_locations:
                pkg_dir = spec.submodule_search_locations[0]
                lib_dir = os.path.join(pkg_dir, "lib")

                if os.path.exists(lib_dir):
                    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
                    os.environ["LD_LIBRARY_PATH"] = (
                        f"{lib_dir}:{current_ld}" if current_ld else lib_dir
                    )

                    for file in os.listdir(lib_dir):
                        if file.endswith(".so") or ".so." in file:
                            try:
                                ctypes.CDLL(os.path.join(lib_dir, file))
                            except OSError:
                                pass
        except Exception:
            pass


def format_timestamp(seconds: float) -> str:
    """Converts seconds into SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    folder_path = Path(target_dir)

    if not folder_path.is_dir():
        print(f"❌ Error: '{target_dir}' is not a valid directory.")
        sys.exit(1)

    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    videos = [
        p
        for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in video_extensions
    ]

    if not videos:
        print(f"⚠️ No video files found in {folder_path.resolve()}")
        sys.exit(0)

    print(f"📂 Found {len(videos)} video(s) in {folder_path.resolve()}")

    # Dynamically locate models/turbo relative to subgen.py location
    script_dir = Path(__file__).parent.resolve()
    local_model_path = script_dir / "models" / "turbo"

    if local_model_path.exists():
        model_path = str(local_model_path)
        print("⏳ Loading local Whisper turbo model into VRAM (int8)...")
    else:
        model_path = "deepdml/faster-whisper-large-v3-turbo-ct2"
        print(f"⚠️ Local model path '{local_model_path}' not found.")
        print("⏳ Falling back to Hugging Face cache...")

    model = WhisperModel(model_path, device="cuda", compute_type="int8")
    print("✅ Model loaded successfully.\n")

    for video_path in videos:
        srt_path = video_path.with_suffix(".srt")

        if srt_path.exists():
            print(f"⏭️  Skipping '{video_path.name}' - SRT already exists.")
            continue

        print(f"🎬 Processing: {video_path.name}")

        try:
            segments, info = model.transcribe(
                str(video_path),
                language="en",
                condition_on_previous_text=False,
            )

            print(f"⏱️  Audio duration: {info.duration:.2f} seconds")

            with open(srt_path, "w", encoding="utf-8") as f:
                with tqdm(
                    total=info.duration,
                    unit=" audio sec",
                    bar_format="{l_bar}{bar:40}{r_bar}",
                ) as pbar:
                    current_time = 0.0

                    for i, segment in enumerate(segments, start=1):
                        start = format_timestamp(segment.start)
                        end = format_timestamp(segment.end)
                        f.write(
                            f"{i}\n{start} --> {end}\n{segment.text.strip()}\n\n"
                        )

                        time_processed = max(0.0, segment.end - current_time)
                        pbar.update(time_processed)
                        current_time = max(current_time, segment.end)

                    if current_time < info.duration:
                        pbar.update(info.duration - current_time)

            print(f"💾 Saved: {srt_path.name}\n")

            print("🌡️ Cooling down for 20 seconds...")
            time.sleep(20)

        except Exception as e:
            print(f"❌ Failed to process '{video_path.name}': {e}")
            if srt_path.exists():
                srt_path.unlink()
            print()

    print("🎉 All videos processed successfully!")


if __name__ == "__main__":
    load_cuda_libs()
    main()
