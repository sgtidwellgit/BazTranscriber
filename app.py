"""Gradio UI for transcribing audio/video files with OpenAI Whisper."""

from pathlib import Path

import gradio as gr
import whisper
from whisper.utils import get_writer

MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
OUTPUT_FORMATS = ["txt", "srt", "vtt", "tsv", "json"]

_loaded_models = {}


def get_model(model_size: str, device: str):
    key = (model_size, device)
    if key not in _loaded_models:
        _loaded_models[key] = whisper.load_model(model_size, device=device)
    return _loaded_models[key]


def transcribe(file_path, model_size, language, device, formats, progress=gr.Progress()):
    if not file_path:
        raise gr.Error("Please select a file first.")
    if not formats:
        raise gr.Error("Select at least one output format.")

    progress(0, desc=f"Loading model '{model_size}'...")
    model = get_model(model_size, device)

    transcribe_kwargs = {}
    language = (language or "").strip()
    if language and language.lower() != "auto":
        transcribe_kwargs["language"] = language

    progress(0.2, desc="Transcribing...")
    result = model.transcribe(file_path, **transcribe_kwargs)

    input_path = Path(file_path)
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
