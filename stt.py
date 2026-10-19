#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "assemblyai>=0.40,<1",
#   "tqdm>=4.66,<5",
#   "faster-whisper>=1.1,<2",
#   "av>=11,<18",  # PyAV 18+ removed metadata_errors used by faster-whisper 1.2.1
#   "mlx-whisper>=0.4,<1; sys_platform == 'darwin' and platform_machine == 'arm64'",
# ]
# ///
"""Transcribe audio/video files using AssemblyAI or local Whisper (automatic GPU backend selection)."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import NamedTuple

AUDIO = frozenset({".opus", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".aiff", ".aif"})
VIDEO = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".mpeg", ".mpg"})
LOG = logging.getLogger("stt")


@contextmanager
def live_stage(name: str):
    """Show one live status line in single-file mode; retain detail in --log-file."""
    # The terminal handler ignores these records because we write the ongoing
    # stage ourselves, allowing `done` to appear on the *same* line.
    LOG.info("%s:", name, extra={"inline_stage": True})
    sys.stderr.write(f"INFO: {name}:")
    sys.stderr.flush()
    started = perf_counter()
    try:
        yield
    except Exception:
        sys.stderr.write(" failed\n")
        sys.stderr.flush()
        LOG.error("%s: failed", name, extra={"inline_stage": True})
        raise
    else:
        elapsed = perf_counter() - started
        sys.stderr.write(f" done ({elapsed:.1f}s)\n")
        sys.stderr.flush()
        LOG.info("%s: done (%.1fs)", name, elapsed, extra={"inline_stage": True})


class Job(NamedTuple):
    original: Path
    audio: Path
    output: Path
    needs_conversion: bool


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="""Transcribe audio/video files using AssemblyAI or local Whisper (automatic GPU backend selection).

For single files:
  - If video (e.g., .mp4, .mkv, .mov), convert to .opus first (unless .opus already exists)
  - Creates a .txt file alongside the .opus (or original audio) file
  - Saves transcription to a .txt file and prints it to stdout

For directories:
  - Recursively finds all matching audio/video files
  - Converts video to opus before transcribing (if no existing opus file)
  - Uses parallel conversion for faster performance
  - Creates .txt files alongside each file
  - Shows a progress bar
  - Skips files that already have .txt versions""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", type=Path,
                        help="Single audio/video file to transcribe (creates .txt and prints transcript)")
    source.add_argument("--directory", type=Path,
                        help="Starting directory for recursive search of audio/video files")
    parser.add_argument("--extension", type=str,
                        help="Specific file extension to search for (e.g., 'opus', 'mkv', 'mp3', 'mov')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without making any API calls or creating files")
    parser.add_argument("--log-file", type=Path,
                        help="Optional log file to store info logs")
    parser.add_argument("--provider", choices=("whisper", "assemblyai"), default="whisper",
                        help="Transcription provider (default: whisper, local; assemblyai uses API)")
    parser.add_argument("--backend", choices=("auto", "mlx", "cuda", "cpu"), default=None,
                        help="Local Whisper backend (default: auto; incompatible with AssemblyAI)")
    parser.add_argument("--model", default="base",
                        help="Local Whisper model size or Hugging Face model ID (default: base)")
    parser.add_argument("--language", default=None, help="Spoken language code, e.g. en (default: auto detect)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed HTTP, model loading, and transcription diagnostics")
    return parser


def make_job(path: Path) -> Job:
    is_video = path.suffix.lower() in VIDEO
    audio = path.with_suffix(".opus") if is_video else path
    return Job(path, audio, path.with_suffix(".txt"), is_video and not audio.is_file())


def discover(folder: Path, extension: str | None) -> list[Job]:
    suffixes = {extension} if extension else AUDIO | VIDEO
    candidates = sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in suffixes), key=lambda p: str(p).casefold())
    # A converted .opus and its source video should not be transcribed twice.
    # Prefer videos over audio files where both write the same transcript path.
    candidates.sort(key=lambda p: p.suffix.lower() not in VIDEO)
    seen_outputs: set[Path] = set()
    jobs: list[Job] = []
    for path in candidates:
        job = make_job(path)
        if job.output in seen_outputs:
            LOG.info("Skipping duplicate transcript target: %s", path)
            continue
        seen_outputs.add(job.output)
        if job.output.is_file():
            LOG.info("Skipping existing transcript: %s", job.output)
            continue
        jobs.append(job)
    return jobs


def convert(job: Job) -> None:
    if not job.needs_conversion or job.audio.exists():
        return
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is not installed or isn't on PATH. Install FFmpeg to convert video files.")
    # Write to a temp file first, then rename: failed conversions must not
    # leave an apparently usable .opus behind.
    fd, temp_name = tempfile.mkstemp(prefix=".stt_", suffix=".opus", dir=job.audio.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        completed = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(job.original), "-vn", "-c:a", "libopus", "-b:a", "32k",
             "-ac", "1", str(temp)],
            text=True, capture_output=True, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"FFmpeg failed for {job.original}: {completed.stderr.strip()}")
        temp.replace(job.audio)
        LOG.debug("Converted %s -> %s", job.original, job.audio)
    finally:
        temp.unlink(missing_ok=True)


def choose_backend(provider: str, requested_backend: str | None) -> str:
    """Select a local backend. No cloud provider is selected without explicit opt-in."""
    if provider == "assemblyai":
        if requested_backend is not None:
            raise ValueError("--backend cannot be used with --provider assemblyai")
        return "assemblyai"
    backend = requested_backend or "auto"
    if backend == "mlx":
        if sys.platform != "darwin" or platform.machine().lower() != "arm64":
            raise RuntimeError("MLX requires native arm64 Python on Apple Silicon macOS")
        return "mlx"
    if backend in ("cuda", "cpu"):
        return backend
    if sys.platform == "darwin" and platform.machine().lower() == "arm64":
        return "mlx"
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception as exc:
        LOG.debug("CUDA detection unavailable: %s", exc)
    return "cpu"


def mlx_repo(model: str) -> str:
    """Resolve common Whisper model names to existing MLX model repositories."""
    if "/" in model or Path(model).exists():
        return model
    names = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large": "mlx-community/whisper-large-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        "turbo": "mlx-community/whisper-large-v3-turbo",
    }
    if model not in names:
        raise ValueError(
            f"Unknown MLX model {model!r}; choose one of: {', '.join(names)}, "
            "or supply an exact Hugging Face repo ID (owner/model)."
        )
    return names[model]


def load_backend(backend: str, model: str):
    if backend == "assemblyai":
        key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("Set ASSEMBLYAI_API_KEY for --provider assemblyai")
        import assemblyai as aai
        aai.settings.api_key = key
        return aai.Transcriber()
    if backend == "mlx":
        import mlx_whisper
        return mlx_whisper
    from faster_whisper import WhisperModel
    return WhisperModel(model, device="cuda" if backend == "cuda" else "cpu",
                        compute_type="float16" if backend == "cuda" else "int8")


def transcribe(job: Job, transcriber: object, backend: str, model: str,
               language: str | None) -> str:
    if backend == "mlx":
        # Do not pass beam_size to MLX: the released decoder only implements
        # greedy decoding, and even beam_size=1 can trigger its unsupported path.
        result = transcriber.transcribe(str(job.audio), path_or_hf_repo=mlx_repo(model),
                                       language=language, verbose=None)
        return result["text"].strip()
    if backend in ("cpu", "cuda"):
        segments, _ = transcriber.transcribe(str(job.audio), beam_size=1, language=language)
        return " ".join(segment.text.strip() for segment in segments).strip()
    import assemblyai as aai
    transcript = transcriber.transcribe(str(job.audio))
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI error: {transcript.error}")
    if transcript.text is None:
        raise RuntimeError("AssemblyAI returned no transcription text")
    return transcript.text


def save_transcript(path: Path, text: str) -> bool:
    fd, temp_name = tempfile.mkstemp(prefix=".stt_", suffix=".txt", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.write("\n")
        # Never overwrite an already-existing transcript (e.g. created during this run).
        if path.exists():
            LOG.info("Transcript appeared during processing; not overwriting: %s", path)
            return False
        Path(temp_name).replace(path)
        return True
    finally:
        Path(temp_name).unlink(missing_ok=True)


def main() -> int:
    parser = args_parser()
    args = parser.parse_args()
    extension = None
    if args.extension:
        extension = "." + args.extension.strip().lower().lstrip(".")
        if extension not in AUDIO | VIDEO:
            parser.error(f"unsupported extension: {args.extension}")
    if args.provider == "assemblyai" and args.backend is not None:
        parser.error("--backend cannot be used with --provider assemblyai")
    if args.file:
        if not args.file.is_file():
            parser.error(f"file not found: {args.file}")
        if args.file.suffix.lower() not in AUDIO | VIDEO:
            parser.error(f"unsupported file format: {args.file.suffix}")
        if extension and args.file.suffix.lower() != extension:
            parser.error("--file does not match --extension")
    else:
        args.directory = args.directory or Path.cwd()
        if not args.directory.is_dir():
            parser.error(f"directory not found: {args.directory}")

    stream_handler = logging.StreamHandler(sys.stderr)
    # Inline status records go into --log-file but must not be printed twice.
    stream_handler.addFilter(lambda record: not getattr(record, "inline_stage", False))
    handlers: list[logging.Handler] = [stream_handler]
    if args.log_file and not args.dry_run:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    # Don't elevate third-party libraries to INFO just because the app reports progress.
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s", handlers=handlers, force=True)
    LOG.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if not args.verbose:
        # Hugging Face / hf-xet can print model-cache download and reconstruction
        # progress even when the model is already cached. Keep normal runs quiet,
        # while allowing full diagnostics with --verbose. This must be set before
        # importing mlx_whisper or huggingface_hub.
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        for noisy in ("httpx", "httpcore", "huggingface_hub", "filelock", "fsspec"):
            logging.getLogger(noisy).setLevel(logging.ERROR)


    jobs = [make_job(args.file)] if args.file else discover(args.directory, extension)
    if not args.file:
        LOG.info("Found %d file(s) to process in %s", len(jobs), args.directory)
    if args.dry_run:
        for job in jobs:
            print(f"{'CONVERT + TRANSCRIBE' if job.needs_conversion else 'TRANSCRIBE'}: {job.original} -> {job.output}")
        print(f"Dry run: {len(jobs)} file(s); provider={args.provider}; backend={args.backend or 'auto'}; no files created or API calls made.")
        return 0
    if not jobs:
        return 0
    from tqdm import tqdm
    try:
        backend = choose_backend(args.provider, args.backend)
        LOG.info("Selected backend: %s (OS=%s, architecture=%s)",
                 backend.upper(), platform.system(), platform.machine())
        try:
            with live_stage(f"Initializing {backend.upper()}"):
                transcriber = load_backend(backend, args.model)
        except Exception as exc:
            if args.provider == "whisper" and args.backend in (None, "auto") and backend in ("mlx", "cuda"):
                LOG.warning("%s unavailable (%s); falling back to CPU", backend.upper(), exc)
                backend = "cpu"
                with live_stage("Initializing CPU"):
                    transcriber = load_backend(backend, args.model)
            else:
                raise
    except Exception as exc:
        LOG.error("Could not initialize transcription backend: %s", exc)
        return 2

    # Single-file mode reports live steps without repeating filenames.
    # Directory mode keeps parallel conversion and per-file progress bars.
    conversion_errors: dict[Path, Exception] = {}
    needs_conversion = [job for job in jobs if job.needs_conversion]
    if args.file:
        if needs_conversion:
            job = needs_conversion[0]
            try:
                with live_stage("Converting to Opus"):
                    convert(job)
            except Exception as exc:
                conversion_errors[job.original] = exc
                LOG.error("Conversion failed: %s: %s", job.original, exc)
    elif needs_conversion:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(convert, job): job for job in needs_conversion}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
                job = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    conversion_errors[job.original] = exc
                    LOG.error("Conversion failed: %s: %s", job.original, exc)

    failures = len(conversion_errors)
    run_start = perf_counter()
    single_file_transcript: str | None = None
    for job in tqdm(jobs, desc="Transcribing", unit="file", disable=bool(args.file)):
        if job.original in conversion_errors:
            continue
        try:
            if not job.audio.is_file():
                raise FileNotFoundError(f"Audio file missing: {job.audio}")
            if args.file:
                with live_stage("Transcribing"):
                    text = transcribe(job, transcriber, backend, args.model, args.language)
            else:
                text = transcribe(job, transcriber, backend, args.model, args.language)
            saved = save_transcript(job.output, text)
            if args.file:
                if saved:
                    LOG.info("Transcript saved: %s", job.output.name)
                else:
                    LOG.info("Transcript already exists: %s (not overwritten)", job.output.name)
                # Print only after the final summary, below.
                single_file_transcript = text
            else:
                LOG.debug("Transcript saved: %s", job.output)
        except Exception as exc:
            failures += 1
            LOG.error("Failed to transcribe %s: %s", job.original, exc)
    LOG.info("Finished: %d successful, %d failed (transcription stage %.1fs)", len(jobs) - failures, failures, perf_counter() - run_start)
    if single_file_transcript is not None:
        print(single_file_transcript, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
