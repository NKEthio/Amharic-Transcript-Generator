#!/usr/bin/env python3
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from amharic_asr.transcribe import transcribe_audio


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
    args = parser.parse_args()

    text = transcribe_audio(args.model_dir, args.audio_path, device=args.device)
    print(text)


if __name__ == "__main__":
    main()
