# Baz Transcriber

Baz Transcriber is a local, privacy-friendly tool for turning audio and video files into text transcripts using [OpenAI's Whisper](https://github.com/openai/whisper) speech recognition model. It runs entirely on your own machine — no files are uploaded to any external service — and comes with two ways to use it: a point-and-click web interface built with [Gradio](https://www.gradio.app/), and a traditional command-line script for scripting or batch use.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Web UI](#running-the-web-ui)
- [Using the Command Line Tool](#using-the-command-line-tool)
- [Choosing a Model Size](#choosing-a-model-size)
- [Output Formats](#output-formats)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Runs locally.** Your media files never leave your machine — everything is processed on your own CPU or GPU.
- **Supports many file types.** Anything [ffmpeg](https://ffmpeg.org/) can decode works as input, including `.mp4`, `.mkv`, `.mov`, `.avi`, `.wav`, `.mp3`, `.m4a`, `.flac`, and `.ogg`.
- **Two interfaces.** Use the Gradio web app for an interactive, visual workflow, or `transcribe.py` for scripted/automated use.
- **Multiple output formats.** Generate `.txt`, `.srt`, `.vtt`, `.tsv`, and `.json` transcripts in a single run.
- **GPU accelerated.** Uses CUDA when available for dramatically faster transcription, with a CPU fallback for machines without a compatible GPU.
- **Configurable model size and language.** Trade off speed against accuracy, and optionally force a specific spoken language instead of relying on auto-detection.

## How It Works

Both the web UI and the command-line tool are thin wrappers around the Whisper Python library:

1. The selected media file is handed to Whisper, which uses `ffmpeg` internally to extract and resample its audio track.
2. Whisper's neural network model processes the audio and produces a transcript, along with timestamped segments.
3. The transcript is written out in whichever formats you choose, and (in the web UI) displayed directly on screen.

No audio or video content is sent anywhere outside your computer — all processing happens locally using the model weights downloaded by Whisper the first time you run a given model size.

## Requirements

- **Python 3.12** (the project was built and tested against this version).
- **[ffmpeg](https://ffmpeg.org/download.html)** installed and available on your system `PATH`. Whisper shells out to `ffmpeg` to decode audio, so transcription will fail without it.
- **(Optional but recommended) An NVIDIA GPU with CUDA support.** Transcription works on CPU as well, but is significantly slower, especially with the larger models.
- **Disk space for model weights.** Whisper downloads model files the first time each size is used; the `large-v3` model is roughly 3 GB.

## Installation

1. Clone the repository:

   ```powershell
   git clone https://github.com/sgtidwellgit/BazTranscriber.git
   cd BazTranscriber
   ```

2. Create a virtual environment:

   ```powershell
   python -m venv whisper_env
   ```

3. Activate it:

   ```powershell
   whisper_env\Scripts\activate
   ```

4. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

   `requirements.txt` installs a CUDA-enabled build of PyTorch (`torch==2.14.0+cu130`) from PyTorch's own package index. If your machine does not have a compatible NVIDIA GPU, install the CPU build of `torch` first (see [pytorch.org](https://pytorch.org/get-started/locally/) for the correct command for your system), and then run `pip install -r requirements.txt` — pip will see `torch` is already satisfied and install the remaining packages.

5. Confirm `ffmpeg` is installed and on your `PATH`:

   ```powershell
   ffmpeg -version
   ```

   If this fails, install ffmpeg (for example, `winget install Gyan.FFmpeg` on Windows) and restart your terminal.

## Running the Web UI

With the virtual environment activated, start the app:

```powershell
python app.py
```

Gradio will print a local URL (typically `http://127.0.0.1:7860`). Open it in your browser. You'll see the following controls:

| Field | Description |
|---|---|
| **Audio / Video File** | Click to browse for, or drag and drop, the file you want to transcribe. |
| **Model** | The Whisper model size to use. Larger models are more accurate but slower and require more memory. See [Choosing a Model Size](#choosing-a-model-size). |
| **Language** | The spoken language in the file, e.g. `English`. Leave as `auto` to let Whisper detect the language automatically from the first 30 seconds of audio. |
| **Device** | `cuda` to use your GPU (recommended if available) or `cpu` to run on the processor only. |
| **Output Formats** | Check which transcript file formats you want written to disk (`txt`, `srt`, `vtt`, `tsv`, `json`). |

Click **Transcribe** to start processing. The first time you use a given model size, Whisper will download its weights, which can take a few minutes depending on your connection. Once complete, the transcript text appears in the **Transcript** box on screen, and the generated files are available for download from the **Download Files** box.

Generated files are also saved locally to a `transcripts/<filename>/` folder next to `app.py`.

The model stays loaded in memory after the first run, so subsequent transcriptions using the same model size and device will start immediately without reloading.

## Using the Command Line Tool

For scripting, automation, or batch processing, use `transcribe.py` directly:

```powershell
python transcribe.py "path\to\media.mp4" --model large-v3 --language English --device cuda
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `input` | *(required)* | Path to the audio or video file to transcribe. |
| `--model` | `large-v3` | Whisper model size. One of `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`. |
| `--language` | *(auto-detect)* | Spoken language in the file (e.g. `English`). Omit to let Whisper detect it automatically. |
| `--device` | `cuda` | Device to run inference on: `cuda` or `cpu`. |
| `--output-dir` | `transcripts` | Base directory transcripts are written into (a subfolder named after the input file is created inside it). |
| `--output-format` | `all` | Which format(s) to write: `txt`, `srt`, `vtt`, `tsv`, `json`, or `all` for every format. |

Run `python transcribe.py --help` at any time to see this list from the tool itself.

### Example

```powershell
python transcribe.py "lecture.mp4" --model medium --language English --device cpu --output-format srt
```

This transcribes `lecture.mp4` using the `medium` model on the CPU, forcing English as the spoken language, and writes only the `.srt` subtitle file to `transcripts\lecture\lecture.srt`. The transcript text is also printed to the terminal when the run finishes.

## Choosing a Model Size

Whisper offers several model sizes, trading accuracy for speed and memory usage:

| Model | Relative Speed | Accuracy | Approximate VRAM |
|---|---|---|---|
| `tiny` | Fastest | Lowest | ~1 GB |
| `base` | Very fast | Low | ~1 GB |
| `small` | Fast | Moderate | ~2 GB |
| `medium` | Moderate | Good | ~5 GB |
| `large` / `large-v2` | Slow | Very good | ~10 GB |
| `large-v3` | Slow | Best available | ~10 GB |

For quick drafts or low-resource machines, start with `small` or `medium`. For the highest-quality transcripts (recommended when accuracy matters most), use `large-v3` on a GPU with sufficient VRAM.

## Output Formats

| Format | Description |
|---|---|
| `.txt` | Plain text transcript, no timestamps. |
| `.srt` | SubRip subtitle format, with numbered, timestamped caption blocks — compatible with most video players and editors. |
| `.vtt` | WebVTT subtitle format, similar to `.srt` but used natively by web `<video>` players. |
| `.tsv` | Tab-separated values with per-segment start time, end time, and text — convenient for spreadsheet analysis. |
| `.json` | The full Whisper result object, including segment-level timestamps, detected language, and token-level metadata. |

## Project Structure

```
BazTranscriber/
├── app.py              # Gradio web UI
├── transcribe.py        # Command-line transcription tool
├── requirements.txt      # Python dependencies
├── LICENSE
├── README.md
├── .gitignore
├── whisper_env/          # Local virtual environment (not tracked in git)
└── transcripts/          # Generated output (not tracked in git)
```

## Troubleshooting

**"CUDA out of memory" errors:** Try a smaller model size, or switch `--device` / the Device dropdown to `cpu`.

**`ffmpeg` not found / decoding errors:** Confirm `ffmpeg -version` works in the same terminal you're running the app from. If it's not recognized, ffmpeg is either not installed or not on your `PATH`.

**Transcription running on CPU even though you have a GPU:** Confirm your PyTorch installation matches your CUDA version, and that `torch.cuda.is_available()` returns `True` in a Python shell within the activated virtual environment.

**First run is very slow:** This is expected — Whisper downloads the selected model's weights the first time it's used. Subsequent runs with the same model size will be much faster since the weights are cached locally.

**Incorrect language detected:** Set the Language field/`--language` argument explicitly instead of relying on auto-detection, especially for short clips or files with background noise.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — the underlying speech recognition model.
- [Gradio](https://www.gradio.app/) — the web UI framework used for the interactive interface.
