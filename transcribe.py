"""Command-line transcription tool built on OpenAI Whisper.

Usage:
    python transcribe.py "path/to/media.mp4" --model large-v3 --language English --device cuda
"""

import argparse
from pathlib import Path

import whisper
from whisper.utils import get_writer

MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
OUTPUT_FORMATS = ["txt", "srt", "vtt", "tsv", "json"]


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe an audio or video file with Whisper.")
    parser.add_argument("input", help="Path to the audio or video file to transcribe.")
    parser.add_argument("--model", default="large-v3", choices=MODEL_SIZES, help="Whisper model size.")
    parser.add_argument("--language", default=None, help="Spoken language (e.g. English). Omit to auto-detect.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run inference on.")
    parser.add_argument(
        "--output-dir",
        default="transcripts",
        help="Base directory to write transcript files into.",
    )
    parser.add_argument(
        "--output-format",
        default="all",
        choices=OUTPUT_FORMATS + ["all"],
        help="Which transcript format(s) to write.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir) / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model '{args.model}' on {args.device}...")
    model = whisper.load_model(args.model, device=args.device)

    transcribe_kwargs = {}
    if args.language:
        transcribe_kwargs["language"] = args.language

    print(f"Transcribing {input_path.name}...")
    result = model.transcribe(str(input_path), **transcribe_kwargs)

    formats = OUTPUT_FORMATS if args.output_format == "all" else [args.output_format]
    for fmt in formats:
        writer = get_writer(fmt, str(output_dir))
        writer(result, str(input_path))
        print(f"Wrote {output_dir / (input_path.stem + '.' + fmt)}")

    print("\n--- Transcript ---\n")
    print(result["text"].strip())


if __name__ == "__main__":
    main()
