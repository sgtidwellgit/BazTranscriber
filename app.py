"""Gradio UI for transcribing audio/video files with OpenAI Whisper."""

import importlib
import logging
import time
import tqdm as tqdm_module
from datetime import datetime
from pathlib import Path

import gradio as gr
import whisper
from whisper.utils import get_writer

# whisper/__init__.py re-exports a `transcribe` function that shadows the
# `whisper.transcribe` submodule attribute, so grab the real module via
# importlib rather than `import whisper.transcribe as ...`.
whisper_transcribe = importlib.import_module("whisper.transcribe")

MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
OUTPUT_FORMATS = ["txt", "srt", "vtt", "tsv", "json"]

_loaded_models = {}

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_console_handler)


def get_model(model_size: str, device: str):
    key = (model_size, device)
    if key not in _loaded_models:
        logger.info(f"Loading model '{model_size}' on {device}...")
        load_start = time.monotonic()
        _loaded_models[key] = whisper.load_model(model_size, device=device)
        logger.info(f"Model loaded in {time.monotonic() - load_start:.1f}s")
    return _loaded_models[key]


def _make_progress_tqdm(progress, logger, file_logger):
    """tqdm subclass that relays Whisper's internal frame-count progress to Gradio and a log file."""

    class ProgressTqdm(tqdm_module.tqdm):
        _last_logged_pct = -1

        def update(self, n=1):
            super().update(n)
            if self.total:
                frac = min(self.n / self.total, 1.0)
                progress(0.2 + frac * 0.75, desc=f"Transcribing... {frac * 100:.0f}%")
                pct = int(frac * 100)
                if pct != self._last_logged_pct and pct % 5 == 0:
                    ProgressTqdm._last_logged_pct = pct
                    file_logger.info(f"Transcribe progress: {pct}% ({self.n}/{self.total} frames)")

    return ProgressTqdm


def transcribe(file_path, model_size, language, device, formats, progress=gr.Progress()):
    if not file_path:
        raise gr.Error("Please select a file first.")
    if not formats:
        raise gr.Error("Select at least one output format.")

    input_path = Path(file_path)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}_{input_path.stem}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"Log file: {log_path}")

    try:
        progress(0, desc=f"Loading model '{model_size}'...")
        model = get_model(model_size, device)

        transcribe_kwargs = {}
        language = (language or "").strip()
        if language and language.lower() != "auto":
            transcribe_kwargs["language"] = language

        progress(0.2, desc="Transcribing...")
        logger.info(f"Transcribing {input_path.name}...")
        transcribe_start = time.monotonic()

        original_tqdm = whisper_transcribe.tqdm.tqdm
        whisper_transcribe.tqdm.tqdm = _make_progress_tqdm(progress, logger, logger)
        try:
            result = model.transcribe(file_path, verbose=False, **transcribe_kwargs)
        finally:
            whisper_transcribe.tqdm.tqdm = original_tqdm

        logger.info(f"Transcription finished in {time.monotonic() - transcribe_start:.1f}s")
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()

    output_dir = Path("transcripts") / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = []
    for fmt in formats:
        writer = get_writer(fmt, str(output_dir))
        writer(result, str(input_path))
        output_files.append(str(output_dir / f"{input_path.stem}.{fmt}"))

    progress(1.0, desc="Done")
    return result["text"].strip(), output_files


with gr.Blocks(title="Baz Transcriber") as demo:
    gr.Markdown("# Baz Transcriber\nSelect an audio or video file to generate a transcript with Whisper.")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Audio / Video File")
            model_dropdown = gr.Dropdown(MODEL_SIZES, value="large-v3", label="Model")
            language_input = gr.Textbox(value="auto", label="Language (e.g. English, or 'auto' to detect)")
            device_dropdown = gr.Dropdown(["cuda", "cpu"], value="cuda", label="Device")
            formats_checkbox = gr.CheckboxGroup(OUTPUT_FORMATS, value=OUTPUT_FORMATS, label="Output Formats")
            run_button = gr.Button("Transcribe", variant="primary")
        with gr.Column():
            transcript_output = gr.Textbox(label="Transcript", lines=20)
            files_output = gr.File(label="Download Files", file_count="multiple")

    run_button.click(
        fn=transcribe,
        inputs=[file_input, model_dropdown, language_input, device_dropdown, formats_checkbox],
        outputs=[transcript_output, files_output],
    )

if __name__ == "__main__":
    demo.launch()
