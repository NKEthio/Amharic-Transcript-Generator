#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from amharic_asr.transcribe import transcribe_audio, load_transcription_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch transcribe Amharic audio files.")
    parser.add_argument("--model-dir", required=True, help="Path to fine-tuned model directory.")
    parser.add_argument("--input", required=True, help="Path to a directory of audio files or a CSV file.")
    parser.add_argument("--output-dir", default="outputs/transcriptions", help="Directory to save transcriptions.")
    parser.add_argument("--audio-column", default="audio_path", help="Column name for audio paths (if input is CSV).")
    parser.add_argument(
        "--format",
        choices=["txt", "srt", "vtt"],
        default="txt",
        help="Output format (txt, srt, or vtt).",
    )
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
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine input files
    audio_files = []
    if os.path.isdir(args.input):
        audio_files = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))
        ]
        audio_files.sort()
    elif args.input.lower().endswith('.csv'):
        df = pd.read_csv(args.input)
        if args.audio_column not in df.columns:
            print(f"Error: CSV must contain column '{args.audio_column}'")
            sys.exit(1)
        audio_files = df[args.audio_column].tolist()
    else:
        print(f"Error: Input must be a directory or a CSV file.")
        sys.exit(1)

    if not audio_files:
        print("No audio files found to process.")
        return

    # Pre-load the pipeline for efficiency
    print(f"Loading model from {args.model_dir}...")
    asr_pipeline = load_transcription_pipeline(
        args.model_dir,
        device=args.device,
        chunk_length_s=args.chunk_length_s
    )

    print(f"Processing {len(audio_files)} files...")
    for audio_path in tqdm(audio_files):
        if not os.path.exists(audio_path):
            print(f"Warning: File not found: {audio_path}")
            continue

        try:
            # Generate transcript
            transcript = transcribe_audio(
                args.model_dir,
                audio_path,
                device=args.device,
                format=args.format,
                chunk_length_s=args.chunk_length_s,
                asr_pipeline=asr_pipeline
            )

            # Save transcript
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(args.output_dir, f"{base_name}.{args.format}")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcript)

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    print(f"\nDone! Transcriptions saved to {args.output_dir}")


if __name__ == "__main__":
    main()
