#!/usr/bin/env python3
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from amharic_asr.transcribe import transcribe_audio, to_srt, to_vtt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Amharic transcript from audio.")
    parser.add_argument("--model-dir", required=True, help="Path to fine-tuned model directory.")
    parser.add_argument("--audio-path", required=True, help="Path to audio file.")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Device index, use -1 for CPU. Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--chunk-length-s",
        type=int,
        default=30,
        help="Chunk size in seconds for long audio transcription.",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "srt", "vtt"],
        default="txt",
        help="Output format (default: txt).",
    )
    parser.add_argument(
        "--output",
        help="Path to save the output. If not provided, prints to stdout.",
    )
    args = parser.parse_args()

    return_timestamps = args.format != "txt"
    result = transcribe_audio(
        args.model_dir,
        args.audio_path,
        device=args.device,
        chunk_length_s=args.chunk_length_s,
        return_timestamps=return_timestamps,
    )

    if args.format == "srt":
        output_text = to_srt(result["chunks"])
    elif args.format == "vtt":
        output_text = to_vtt(result["chunks"])
    else:
        output_text = result["text"]

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Transcript saved to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
