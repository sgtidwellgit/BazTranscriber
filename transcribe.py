"""Command-line transcription tool built on OpenAI Whisper.

Usage:
    python transcribe.py "path/to/media.mp4" --model large-v3 --language English --device cuda
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import whisper
from whisper.utils import get_writer

MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
OUTPUT_FORMATS = ["txt", "srt", "vtt", "tsv", "json"]


def setup_logging(input_path: Path) -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}_{input_path.stem}.log"

    logger = logging.getLogger("transcribe")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Log file: {log_path}")
    return logger


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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-segment transcription output (only log timing/status).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger = setup_logging(input_path)

    output_dir = Path(args.output_dir) / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model '{args.model}' on {args.device}...")
    load_start = time.monotonic()
    model = whisper.load_model(args.model, device=args.device)
    logger.info(f"Model loaded in {time.monotonic() - load_start:.1f}s")

    transcribe_kwargs = {}
    if args.language:
        transcribe_kwargs["language"] = args.language

    logger.info(f"Transcribing {input_path.name}...")
    transcribe_start = time.monotonic()
    result = model.transcribe(str(input_path), verbose=not args.quiet, **transcribe_kwargs)
    logger.info(f"Transcription finished in {time.monotonic() - transcribe_start:.1f}s")

    formats = OUTPUT_FORMATS if args.output_format == "all" else [args.output_format]
    for fmt in formats:
        writer = get_writer(fmt, str(output_dir))
        writer(result, str(input_path))
        logger.info(f"Wrote {output_dir / (input_path.stem + '.' + fmt)}")

    print("\n--- Transcript ---\n")
    print(result["text"].strip())


if __name__ == "__main__":
    main()
