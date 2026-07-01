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
    parser.add_argument("--audio-path", required=True, help="Path to audio file or directory.")
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
        help="Output format (txt, srt, or vtt).",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task to perform: transcribe or translate.",
    )
    parser.add_argument(
        "--output",
        help="Path to save the transcript. If not provided, prints to stdout (only for single file).",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save transcripts (for batch processing).",
    )
    args = parser.parse_args()

    audio_files = []
    if os.path.isdir(args.audio_path):
        extensions = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
        for root, _, files in os.walk(args.audio_path):
            for file in files:
                if file.lower().endswith(extensions):
                    audio_files.append(os.path.join(root, file))
        print(f"Found {len(audio_files)} audio files in {args.audio_path}")
    else:
        audio_files = [args.audio_path]

    for audio_file in audio_files:
        print(f"Processing {audio_file}...")
        try:
            text = transcribe_audio(
                args.model_dir,
                audio_file,
                device=args.device,
                chunk_length_s=args.chunk_length_s,
                format=args.format,
                task=args.task,
            )

            if len(audio_files) > 1 or args.output_dir:
                # Batch mode or explicit output-dir
                out_dir = args.output_dir or os.path.join(os.path.dirname(audio_file), "transcripts")
                os.makedirs(out_dir, exist_ok=True)

                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                out_path = os.path.join(out_dir, f"{base_name}.{args.format}")

                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Transcript saved to {out_path}")
            elif args.output:
                # Single file mode with explicit output path
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Transcript saved to {args.output}")
            else:
                # Single file mode to stdout
                print(text)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")


if __name__ == "__main__":
    main()
