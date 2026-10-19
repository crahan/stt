# STT — universal local Whisper / AssemblyAI (`uv` script)

A single-file transcription CLI with **PEP 723** dependencies inside `stt.py`. Run it with `./stt.py` or `uv run stt.py`.

## Automatic hardware choice

`--provider whisper` (the default) uses **local transcription**. `--backend auto` (the implicit default) detects the backend at startup:

| Platform | Selected engine |
| --- | --- |
| Native Apple Silicon macOS (`arm64`) | `mlx-whisper` using Apple's MLX/Metal backend |
| Linux/Windows with supported NVIDIA CUDA and CTranslate2 | `faster-whisper` on CUDA |
| Others, including Intel Macs | `faster-whisper` on CPU |

When automatic startup of an accelerated backend fails, it attempts to fall back to CPU. Forced backends do **not** silently fall back. NVIDIA GPU detection does not guarantee that the installed CTranslate2 and system CUDA/cuDNN libraries are compatible; automatic mode handles *initialization* errors, but an error during actual transcription may require forcing `--backend cpu`.

**AssemblyAI is never selected automatically.** It requires explicit `--provider assemblyai` plus `ASSEMBLYAI_API_KEY`.

## Install

Install [`uv`](https://docs.astral.sh/uv/) and [FFmpeg](https://ffmpeg.org/). On macOS: `brew install uv ffmpeg`. Python packages install automatically on first `uv run`; MLX Whisper is declared with an Apple Silicon-specific dependency marker so it is not installed on other machines.

Use a native Apple Silicon terminal/Python (not Rosetta) to enable MLX. `uname -m` should say `arm64`.

## Options

- `--provider whisper|assemblyai` chooses local or cloud transcription. Default: `whisper`.
- `--backend auto|mlx|cuda|cpu` chooses local Whisper acceleration. Default: `auto`. This flag is invalid with AssemblyAI.

## Examples

```bash
# Fast local transcription, automatically uses best supported backend
./stt.py --file recording.webm --model small

# Explicit Apple MLX
./stt.py --backend mlx --model small --file recording.webm

# Always use CPU
./stt.py --backend cpu --model small --file recording.webm

# Existing faster-whisper path (CUDA when available, otherwise CPU)
./stt.py --provider whisper --backend auto --model small --file recording.webm

# English-only audio can avoid language detection
./stt.py --model small --language en --file recording.webm

# Directory and preview
./stt.py --directory ./recordings --model small
./stt.py --directory ./recordings --dry-run

# AssemblyAI cloud (requires API key)
export ASSEMBLYAI_API_KEY='your_api_key'
./stt.py --provider assemblyai --file recording.webm
```

## Decoding defaults

For local Whisper, the app uses greedy decoding by default. With the MLX backend, the script **does not send a `beam_size` argument** because MLX Whisper does not implement beam search. The faster-whisper CPU/CUDA backend internally uses `beam_size=1`. There is no `--beam-size` command-line option.

## Model names and caches

`--model small` loads `mlx-community/whisper-small-mlx` under MLX and `small` under faster-whisper. Similarly for `tiny`, `base`, `medium`, and `large-v3`. `--model large-v3-turbo` is another option. You may supply a Hugging Face repo name or local path for MLX, or a compatible model path for faster-whisper. A model cached under faster-whisper is **not interchangeable** with MLX weights; the first MLX use will download the MLX-specific model. MLX model downloads use the Hugging Face cache by default.

## Output and requirements

Videos are converted to sibling `.opus` files with FFmpeg; existing `.opus` is reused. `.txt` transcripts are saved beside the original file; single-file mode prints the transcript to stdout after the final INFO summary; directory mode does not. Recursive directory mode skips existing `.txt` files and displays a progress bar. `--verbose` enables debugging and Hugging Face model-cache progress; by default only useful stage/completion logs and elapsed times are shown on stderr. `--log-file` saves completed stage messages and errors; `--dry-run` does not download models, convert, or transcribe.

`uv` chooses package versions from the PEP 723 header. `av>=11,<18` is constrained to avoid the `metadata_errors` API regression seen with newer PyAV and older `faster-whisper`.

To troubleshoot, check `./stt.py --help`, `uname -m`, and `./stt.py --backend cpu --file yourfile.opus`.

## Terminal output

For a single file requiring Opus conversion, each operation (including backend initialization) prints its status immediately
and appends `done` with the elapsed time on **the same line**. Filenames appear
when confirming where the transcript was saved, not in every progress line.

```text
INFO: Selected backend: MLX (OS=Darwin, architecture=arm64)
INFO: Initializing MLX: done (0.5s)
INFO: Converting to Opus: done (25.6s)
INFO: Transcribing: done (31.1s)
INFO: Transcript saved: video.txt
INFO: Finished: 1 successful, 0 failed (transcription stage 31.1s)
```

The transcript itself is also written to stdout in single-file mode, **after**
the `INFO: Finished` message; status messages go to stderr. If the `.opus` file already exists,
the conversion stage is skipped. Directory mode displays progress bars and
saves transcripts without printing their contents.
