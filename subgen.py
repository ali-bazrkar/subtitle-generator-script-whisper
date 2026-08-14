# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "faster-whisper",
#     "huggingface-hub",
#     "tqdm",
#     "nvidia-cublas-cu12",
#     "nvidia-cudnn-cu12",
#     "prompt-toolkit",
# ]
# ///

import argparse
import ctypes
import importlib.util
import os
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from tqdm import tqdm

from prompt_toolkit import prompt

# ANSI Escape Sequences for Terminal Colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

VALID_LANGUAGES = {
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br",
    "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu",
    "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr",
    "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km",
    "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk",
    "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc",
    "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl",
    "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th",
    "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh", "yue"
}

# OpenAI Whisper model preset mappings
PRESETS = {
    # English-only Models
    "tiny.en": {
        "local_folder": "tiny.en",
        "fallback": "Systran/faster-whisper-tiny.en",
        "label": "Whisper Tiny (English-only)",
        "default_cooldown": 5,
    },
    "base.en": {
        "local_folder": "base.en",
        "fallback": "Systran/faster-whisper-base.en",
        "label": "Whisper Base (English-only)",
        "default_cooldown": 10,
    },
    "small.en": {
        "local_folder": "small.en",
        "fallback": "Systran/faster-whisper-small.en",
        "label": "Whisper Small (English-only)",
        "default_cooldown": 20,
    },
    "medium.en": {
        "local_folder": "medium.en",
        "fallback": "Systran/faster-whisper-medium.en",
        "label": "Whisper Medium (English-only)",
        "default_cooldown": 30,
    },

    # Multilingual Models
    "tiny": {
        "local_folder": "tiny",
        "fallback": "Systran/faster-whisper-tiny",
        "label": "Whisper Tiny (Multilingual)",
        "default_cooldown": 5,
    },
    "base": {
        "local_folder": "base",
        "fallback": "Systran/faster-whisper-base",
        "label": "Whisper Base (Multilingual)",
        "default_cooldown": 10,
    },
    "small": {
        "local_folder": "small",
        "fallback": "Systran/faster-whisper-small",
        "label": "Whisper Small (Multilingual)",
        "default_cooldown": 20,
    },
    "medium": {
        "local_folder": "medium",
        "fallback": "Systran/faster-whisper-medium",
        "label": "Whisper Medium (Multilingual)",
        "default_cooldown": 30,
    },
    "turbo": {
        "local_folder": "turbo",
        "fallback": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "label": "Whisper Large-v3 Turbo (Multilingual)",
        "default_cooldown": 40,
    },
    "large-v1": {
        "local_folder": "large-v1",
        "fallback": "Systran/faster-whisper-large-v1",
        "label": "Whisper Large v1 (Multilingual)",
        "default_cooldown": 60,
    },
    "large-v2": {
        "local_folder": "large-v2",
        "fallback": "Systran/faster-whisper-large-v2",
        "label": "Whisper Large v2 (Multilingual)",
        "default_cooldown": 60,
    },
    "large-v3": {
        "local_folder": "large-v3",
        "fallback": "Systran/faster-whisper-large-v3",
        "label": "Whisper Large v3 (Multilingual)",
        "default_cooldown": 60,
    },
}

# Alias mappings
ALIASES = {
    "large": "large-v3",
}


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


def format_duration(seconds: float, force_hours: bool = False) -> str:
    """Formats seconds into MM:SS or HH:MM:SS for display."""
    total_sec = max(0, int(seconds))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60

    if hours > 0 or force_hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def calculate_dynamic_cooldown(
    base_cooldown: int, duration_seconds: float, max_ref_seconds: float = 600.0
) -> int:
    if base_cooldown <= 0:
        return 0

    scaled = int(base_cooldown * (duration_seconds / max_ref_seconds))
    min_floor = min(5, base_cooldown)
    return max(min_floor, min(base_cooldown, scaled))


# --- INTERACTIVE PROMPT HELPERS ---


def prompt_directory_interactive(default_dir: str) -> str:
    """Prompts for target video directory, validates existence, and ensures video files are present."""
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

    print("\n📂 Target Video Directory:")
    print(f"   Current default: {Path(default_dir).resolve()}")

    while True:

        user_input = prompt(f"\nEnter target directory path [default: '{default_dir}']: ").strip()

        # Handle default choice (empty input) vs user input
        raw_path = default_dir if not user_input else user_input.strip("'\"")
        target_path = Path(raw_path)

        # 1. Validate directory existence
        if not target_path.is_dir():
            print(f"\n{RED}❌ Error: Directory '{target_path}' does not exist. Please try again.{RESET}")
            continue

        # 2. Validate video file presence
        videos = [
            p for p in target_path.iterdir()
            if p.is_file() and p.suffix.lower() in video_extensions
        ]

        if not videos:
            print(f"\n{YELLOW}⚠️ Warning: No video files (.mp4, .mkv, .avi, .mov, .webm) found in '{target_path.resolve()}'.{RESET}")
            print(f"{YELLOW}   Please enter a directory that contains supported video files.{RESET}")
            continue

        return str(raw_path)


def prompt_english_interactive():
    """Prompts whether the audio is English."""
    print("\n🌐 Is the audio language English?")

    while True:
        is_eng_input = prompt(
            "\nEnter [Y/n] (default: Y): "
        ).strip().lower()

        if not is_eng_input or is_eng_input == "y":
            return True
        elif is_eng_input == "n":
            return False
        else:
            print(
                f"\n{RED}❌ Invalid input. "
                f"Please enter 'Y' for Yes or 'n' for No.{RESET}"
            )


def prompt_task_interactive():
    """Prompts for transcribe vs translate."""
    print("\n-------------------------------------------------")
    print("\n🗣️  Choose operation for non-English audio:")
    print("  [1] Transcribe (Generate SRT in the original language)")
    print("  [2] Translate (Translate from original language to English SRT)")

    while True:
        task_input = prompt(
            "\nEnter choice [1/2] (default: 1): "
        ).strip()

        if not task_input or task_input == "1":
            return "transcribe"
        elif task_input == "2":
            return "translate"
        else:
            print(
                f"\n{RED}❌ Invalid choice. "
                f"Please enter '1' or '2'.{RESET}"
            )


def prompt_language_interactive():
    """Prompts for and validates the source language code."""
    print("\n-------------------------------------------------")
    print("\n🌍 Source Language Code:")
    print("   Examples: 'fa' (Persian), 'es' (Spanish), 'auto' (Auto-detect)")

    while True:
        lang_input = prompt(
            "\nEnter ISO language code [default: 'auto']: "
        ).strip().lower()

        if not lang_input or lang_input == "auto":
            return "auto"
        elif lang_input in VALID_LANGUAGES:
            return lang_input
        else:
            print(
                f"\n{RED}❌ Invalid language code. "
                f"Please enter 'auto' or a valid ISO language code.{RESET}"
            )


def prompt_language_and_task_interactive():
    """Prompts for language, task, and source language when needed."""
    is_english = prompt_english_interactive()

    if is_english:
        return True, "en", "transcribe"

    task = prompt_task_interactive()
    lang = prompt_language_interactive()

    if lang == "en":
        return True, "en", "transcribe"

    return False, lang, task


def prompt_model_interactive(is_english: bool, task: str = "transcribe", default_model: str = "turbo") -> str:
    """Interactive CLI menu to pick an OpenAI model preset filtered by language type and task."""

    # 1. Filter options: Turbo is excluded for translation tasks
    if is_english:
        options = ["tiny.en", "base.en", "small.en", "medium.en", "turbo", "large-v3"]
    else:
        if task == "translate":
            options = ["tiny", "base", "small", "medium", "large-v3"]
        else:
            options = ["tiny", "base", "small", "medium", "turbo", "large-v3"]

    # 2. Automatically adjust default_model if default ('turbo') is not in available options
    if default_model not in options:
        if not is_english and task == "translate":
            default_model = "small"
        else:
            default_model = "base.en" if is_english else "base"

    print("\n🤖 Choose an OpenAI Whisper model preset:")
    print("---------------------------------------------")
    for idx, key in enumerate(options, 1):
        preset = PRESETS[key]
        print(f"  [{idx:2d}] {key:<10} -> {preset['label']}")
    print("---------------------------------------------")

    while True:
        selection = prompt(f"\nEnter number or model key (default: [{default_model}]): ").strip()

        # Handle empty input (press Enter) -> Return default
        if not selection:
            return default_model

        # Resolve keys/aliases
        resolved_key = ALIASES.get(selection, selection)

        # If it's a valid option for this mode, return it
        if resolved_key in options:
            return resolved_key

        # Explicit feedback if user tries to manually force 'turbo' on a translation task
        if resolved_key in ["turbo", "large-v3-turbo"] and task == "translate":
            print(f"\n{RED}❌ Whisper Turbo does not support translation tasks. Please select a multilingual model.{RESET}")
            continue

        # Handle numeric index input
        try:
            choice_idx = int(selection) - 1
            if 0 <= choice_idx < len(options):
                return options[choice_idx]
            else:
                print(f"\n{RED}❌ Number out of range. Please enter 1-{len(options)}.{RESET}")
        except ValueError:
            print(f"\n{RED}❌ Invalid input. Please enter a valid menu number or model key.{RESET}")


def prompt_device_interactive(default_device: str = "cuda") -> str:
    """Interactive CLI menu to choose target execution device."""
    print("\n⚡  Choose compute device:")
    print("  [1] cuda  (NVIDIA GPU Acceleration)")
    print("  [2] cpu   (CPU Processing)")

    while True:
        selection = prompt(f"\nEnter choice [1/2] (default: [{default_device}]): ").strip().lower()

        if not selection:
            return default_device

        if selection in ["1", "cuda"]:
            return "cuda"
        elif selection in ["2", "cpu"]:
            return "cpu"
        else:
            print(f"\n{RED}❌ Invalid choice. Please enter '1' for cuda or '2' for cpu.{RESET}")


def prompt_compute_type_interactive(default_compute: str = "int8") -> str:
    """Prompts for CTranslate2 compute quantization precision."""
    valid_computes = ["int8", "float16", "int8_float16", "float32"]

    print("\n⚙️  CTranslate2 Compute Type:")
    print(f"   Options: {', '.join(valid_computes)}")

    while True:
        user_input = prompt(f"\nEnter compute precision type [default: '{default_compute}']: ").strip()

        if not user_input:
            return default_compute

        if user_input in valid_computes:
            return user_input
        else:
            print(f"\n{RED}❌ Invalid compute type. Please choose exactly from: {', '.join(valid_computes)}{RESET}")


def prompt_cooldown_interactive(default_cooldown: int | None) -> int | None:
    """Prompts for GPU cooldown duration in seconds."""
    default_str = str(default_cooldown) if default_cooldown is not None else "Preset Default"

    print("\n🌡️  GPU Cooldown Override:")
    print("  Enter base cooldown in seconds (or press Enter to use model defaults)")

    while True:
        user_input = prompt(f"\nEnter cooldown seconds [default: {default_str}]: ").strip()

        if not user_input:
            return default_cooldown

        try:
            val = int(user_input)
            if val >= 0:
                return val
            else:
                print(f"\n{RED}❌ Cooldown cannot be negative. Please enter a valid number.{RESET}")
        except ValueError:
            print(f"\n{RED}❌ Invalid input. Please enter a valid integer.{RESET}")


# --- PARSER AND MAIN execution ---


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SRT subtitles for videos using faster-whisper."
    )
    parser.add_argument(
        "target_dir",
        nargs="?",
        default=".",
        help="Target directory containing video files (default: current directory).",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="turbo",
        help="Model preset ('tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'turbo', 'large-v3') or HF path.",
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Compute device: 'cuda' (NVIDIA GPU) or 'cpu' (default: 'cuda').",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactively configure target directory and all options before running.",
    )
    parser.add_argument(
        "-c",
        "--cooldown",
        type=int,
        default=None,
        help="Cooldown time in seconds between files.",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="auto",
        help="Language code for transcription, e.g., 'en' (default: 'auto').",
    )
    parser.add_argument(
        "-t",
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task to perform: 'transcribe' or 'translate' (default: 'transcribe').",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Compute type for CTranslate2 (default: 'int8').",
    )
    return parser.parse_args()


def get_hf_model(repo_id: str) -> str:
    """Downloads/verifies HF model with progress bars, falling back to local cache if offline."""
    print(f"📥 Checking Hugging Face Hub for '{repo_id}'...")
    try:
        model_path = snapshot_download(repo_id=repo_id)
        print(f"\n{GREEN}✅ Download/verification complete!{RESET}")
        print(f"{GREEN}📍 Cached location: {model_path}{RESET}\n")
        return model_path
    except Exception as err:
        print(f"\n{YELLOW}⚠️ Could not reach Hugging Face Hub ({err}).{RESET}")
        print(f"{YELLOW}🔄 Falling back to local offline cache...{RESET}")
        try:
            model_path = snapshot_download(repo_id=repo_id, local_files_only=True)
            print(f"\n{GREEN}✅ Loaded from local cache (offline mode)!{RESET}")
            print(f"{GREEN}📍 Cached location: {model_path}{RESET}\n")
            return model_path
        except Exception as cache_err:
            print(f"\n{RED}❌ Failed to load model offline. Model is not cached locally: {cache_err}{RESET}")
            sys.exit(1)


def resolve_model_and_cooldown(model_arg: str, cooldown_arg: int | None, script_dir: Path):
    """Resolves model location (project local or HF cache) and cooldown duration."""
    resolved_key = ALIASES.get(model_arg, model_arg)

    if resolved_key in PRESETS:
        preset = PRESETS[resolved_key]
        local_path = script_dir / "models" / preset["local_folder"]
        cooldown = cooldown_arg if cooldown_arg is not None else preset["default_cooldown"]

        if local_path.exists():
            model_path = str(local_path)
            print(f"⏳ Loading local {preset['label']} model from project directory: '{local_path}'...")
        else:
            repo_id = preset["fallback"]
            print(f"{YELLOW}⚠️ Project model directory '{local_path}' not found.{RESET}")
            model_path = get_hf_model(repo_id)
    else:
        cooldown = cooldown_arg if cooldown_arg is not None else 10
        local_target = Path(model_arg)

        if local_target.exists():
            model_path = str(local_target)
            print(f"⏳ Loading model from path: '{model_path}'...")
        else:
            model_path = get_hf_model(model_arg)

    return model_path, cooldown


def main():
    args = parse_args()

    target_dir = args.target_dir
    selected_device = args.device.strip().lower()

    selected_model = args.model.strip()
    model_key = selected_model.lower()
    if model_key in PRESETS or model_key in ALIASES:
        selected_model = model_key

    selected_lang = args.language.strip().lower()
    selected_task = args.task.strip().lower()
    selected_compute = args.compute_type.strip().lower()
    selected_cooldown = args.cooldown

    # --- NON-INTERACTIVE PRE-CHECKS & FALLBACKS ---
    if not args.interactive:
        video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

        # 0. Validate Target Directory & Video Presence
        dir_path = Path(target_dir)
        has_videos = False
        if dir_path.is_dir():
            has_videos = any(
                p.is_file() and p.suffix.lower() in video_extensions
                for p in dir_path.iterdir()
            )

        if not dir_path.is_dir() or not has_videos:
            if not dir_path.is_dir():
                print(f"\n{RED}❌ Error: Directory '{target_dir}' does not exist.{RESET}")
            else:
                print(f"\n{YELLOW}⚠️ Warning: No video files found in '{dir_path.resolve()}'.{RESET}")

            target_dir = prompt_directory_interactive(target_dir)
            print("\n====================================================")

        # 1. Validate Language
        if selected_lang != "auto" and selected_lang not in VALID_LANGUAGES:
            print(
                f"\n{RED}❌ Invalid language code: '{selected_lang}'. "
                f"Please enter a valid Whisper language code.{RESET}"
            )
            selected_lang = prompt_language_interactive()
            print("\n====================================================")

        # 2. Validate Task
        if selected_task not in ["transcribe", "translate"]:
            print(
                f"\n{RED}❌ Invalid task choice: '{selected_task}'. "
                f"Please select 'transcribe' or 'translate'.{RESET}"
            )
            selected_task = prompt_task_interactive()
            print("\n====================================================")

        # 3. CLI Turbo + Translate Conflict Check
        if selected_task == "translate" and selected_model in ["turbo", "large-v3-turbo"]:
            print(
                f"\n{YELLOW}⚠️ Warning: Whisper 'turbo' does not support "
                f"translation tasks.{RESET}"
            )
            print("Please choose a compatible multilingual model:")

            is_eng = selected_lang == "en"
            selected_model = prompt_model_interactive(
                is_english=is_eng,
                task=selected_task,
                default_model="small"
            )
            print("\n====================================================")

        # 4. Language + '.en' model mismatch check
        if selected_lang not in ["en", "auto"] and selected_model.endswith(".en"):
            new_model = selected_model.removesuffix(".en")

            print(
                f"\n{YELLOW}⚠️ Warning: Non-English language "
                f"'{selected_lang}' requested with English-only model "
                f"'{selected_model}'.{RESET}"
            )
            print(
                f"🔄 Automatically swapping to multilingual model "
                f"'{new_model}'.\n"
            )
            selected_model = new_model
            print("\n====================================================")

        # 5. Validate Model Name / Preset / Path
        resolved_key = ALIASES.get(selected_model, selected_model)
        if (
            resolved_key not in PRESETS
            and not Path(selected_model).exists()
            and "/" not in selected_model
        ):
            print(f"\n{RED}❌ Unrecognized model preset or path: '{selected_model}'.{RESET}")
            is_eng = selected_lang == "en"
            selected_model = prompt_model_interactive(
                is_english=is_eng,
                task=selected_task,
                default_model="turbo"
            )
            print("\n====================================================")

        # 6. Validate Compute Device
        if selected_device not in ["cuda", "cpu"]:
            print(f"\n{RED}❌ Invalid device specified: '{selected_device}'.{RESET}")
            selected_device = prompt_device_interactive(default_device="cuda")
            print("\n====================================================")

        # 7. Validate Compute Type
        valid_computes = ["int8", "float16", "int8_float16", "float32"]
        if selected_compute not in valid_computes:
            print(f"\n{RED}❌ Invalid compute precision type: '{selected_compute}'.{RESET}")
            selected_compute = prompt_compute_type_interactive(default_compute="int8")
            print("\n====================================================")

        # 8. Validate Cooldown Value
        if selected_cooldown is not None and selected_cooldown < 0:
            print(f"\n{RED}❌ Cooldown value cannot be negative: {selected_cooldown}.{RESET}")
            selected_cooldown = prompt_cooldown_interactive(default_cooldown=None)
            print("\n====================================================")

    # Step-by-step interactive workflow when -i flag is explicitly set
    else:
        print("\n====================================================")
        print("       🎛️ Interactive Configuration Mode ")
        print("====================================================")

        # Will loop internally until a valid directory WITH video files is selected
        target_dir = prompt_directory_interactive(target_dir)

        print("\n====================================================")
        is_english, selected_lang, selected_task = prompt_language_and_task_interactive()
        print("\n====================================================")

        # Passes task to enforce disabling Turbo for translation
        selected_model = prompt_model_interactive(
            is_english=is_english,
            task=selected_task,
            default_model=selected_model
        )

        print("\n====================================================")
        selected_device = prompt_device_interactive(selected_device)
        print("\n====================================================")
        selected_compute = prompt_compute_type_interactive(selected_compute)
        print("\n====================================================")
        selected_cooldown = prompt_cooldown_interactive(selected_cooldown)
        print("\n====================================================\n")

    folder_path = Path(target_dir)

    # Populate the list of video files to process
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    videos = [
        p
        for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in video_extensions
    ]

    print(f"📂 Found {len(videos)} video(s) in {folder_path.resolve()}")

    if selected_device == "cuda":
        load_cuda_libs()

    script_dir = Path(__file__).parent.resolve()
    model_path, cooldown = resolve_model_and_cooldown(selected_model, selected_cooldown, script_dir)

    print(f"🚀 Initializing WhisperModel on device='{selected_device}' (compute_type='{selected_compute}')...")
    model = WhisperModel(model_path, device=selected_device, compute_type=selected_compute)
    print(f"{GREEN}✅ Model loaded successfully into memory.{RESET}\n")

    unprocessed_videos = [v for v in videos if not v.with_suffix(".srt").exists()]
    total_to_process = len(unprocessed_videos)
    processed_count = 0
    failed_count = 0

    for video_path in videos:
        srt_path = video_path.with_suffix(".srt")

        if srt_path.exists():
            print(f"⏭️ Skipping '{video_path.name}' - SRT already exists.")
            continue

        processed_count += 1
        is_last_video = (processed_count == total_to_process)

        print(f"🎬 Processing: {video_path.name} | Task: {selected_task}")
        success = False

        try:
            # Set language to None if "auto" is passed, allowing faster-whisper to detect it
            transcribe_lang = None if selected_lang == "auto" else selected_lang

            segments, info = model.transcribe(
                str(video_path),
                language=transcribe_lang,
                task=selected_task,
                condition_on_previous_text=False,
            )

            # Runtime Check: Warn if auto-detected non-English on an English-only model
            if selected_model.endswith(".en") and info.language != "en":
                print(f"{YELLOW}⚠️  Warning: Auto-detected language '{info.language}' (p={info.language_probability:.2f}), "
                      f"but using English-only model '{selected_model}'.{RESET}")
                print(f"{YELLOW}   Subtitles may contain gibberish or incorrect transcriptions. "
                      f"Consider using a multilingual model (e.g., 'small' or 'turbo').{RESET}")

            raw_duration = getattr(info, 'duration', 0)
            total_seconds = max(round(raw_duration, 2), 0.1)

            has_hours = total_seconds >= 3600
            total_time_str = format_duration(total_seconds, force_hours=has_hours)

            print(f"⏱️ Audio duration: {total_time_str}")
            if transcribe_lang is None:
                print(f"🎙️ Detected language: '{info.language}' (Probability: {info.language_probability:.2f})")

            with open(srt_path, "w", encoding="utf-8") as f:
                with tqdm(
                    total=total_seconds,
                    unit="s",
                    bar_format="{percentage:3.0f}%|{bar:30}| {desc} [{elapsed}<{remaining}, {rate_fmt}]",
                    dynamic_ncols=True,
                ) as pbar:
                    for i, segment in enumerate(segments, start=1):
                        start = format_timestamp(segment.start)
                        end = format_timestamp(segment.end)
                        f.write(
                            f"{i}\n{start} --> {end}\n{segment.text.strip()}\n\n"
                        )

                        # Clamp bounds & maintain monotonic forward progress
                        seg_end_ms = int(round(segment.end * 1000))
                        target_sec = max(pbar.n, min(seg_end_ms / 1000, total_seconds))

                        # Update delta so tqdm speed/remaining estimates stay accurate
                        delta = round(target_sec - pbar.n, 2)
                        if delta > 0:
                            pbar.update(delta)

                        pbar.set_description_str(
                            f"{format_duration(pbar.n, force_hours=has_hours)}/{total_time_str}",
                            refresh=False,
                        )

                    # Snap to 100% on completion
                    if pbar.n < total_seconds:
                        pbar.update(total_seconds - pbar.n)
                    pbar.set_description_str(
                        f"{total_time_str}/{total_time_str}",
                        refresh=False,
                    )
                    pbar.refresh()

            success = True
            print(f"💾 Saved: {srt_path.name}\n")

            effective_cooldown = calculate_dynamic_cooldown(cooldown, total_seconds)

            if effective_cooldown > 0 and not is_last_video:
                print(f"{BLUE}🌡️ Cooling down for {effective_cooldown}s{RESET}")
                time.sleep(effective_cooldown)
                print()

        except KeyboardInterrupt:
            print(f"{RED}🛑 Interrupted by user while processing '{video_path.name}'.{RESET}")
            sys.exit(130)
        except Exception as e:
            failed_count += 1
            print(f"{RED}❌ Failed to process '{video_path.name}': {e}{RESET}\n")
        finally:
            if not success:
                srt_path.unlink(missing_ok=True)

    if failed_count == 0:
        print(f"{GREEN}🎉 All videos processed successfully! ({processed_count} videos){RESET}")
    else:
        print(
            f"{YELLOW}⚠️ Processing finished: "
            f"{processed_count - failed_count} succeeded, "
            f"{failed_count} failed.{RESET}"
        )


if __name__ == "__main__":
    main()
