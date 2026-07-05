#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm
import jiwer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from amharic_asr.transcribe import transcribe_audio, load_transcription_pipeline
from amharic_asr.data import normalize_amharic


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Amharic ASR model.")
    parser.add_argument("--model-dir", required=True, help="Path to fine-tuned model directory.")
    parser.add_argument("--test-csv", required=True, help="Path to test CSV file.")
    parser.add_argument("--output-report", default="evaluation_report.csv", help="Path to save the evaluation report.")
    parser.add_argument("--audio-column", default="audio_path", help="Column name for audio paths.")
    parser.add_argument("--text-column", default="transcript", help="Column name for reference transcripts.")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Device index, use -1 for CPU. Defaults to CUDA if available.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.test_csv):
        print(f"Error: Test CSV not found at {args.test_csv}")
        sys.exit(1)

    df = pd.read_csv(args.test_csv)
    if args.audio_column not in df.columns or args.text_column not in df.columns:
        print(f"Error: CSV must contain columns '{args.audio_column}' and '{args.text_column}'")
        sys.exit(1)

    results = []

    print(f"Evaluating model from {args.model_dir} on {len(df)} samples...")

    # Load the model once to be reused for all samples
    asr_pipeline = load_transcription_pipeline(
        model_dir=args.model_dir,
        device=args.device
    )

    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row[args.audio_column]
        reference = row[args.text_column]

        # Ensure audio path is absolute or relative to current dir
        if not os.path.isabs(audio_path):
            # Try to find it relative to the CSV file location if not found directly
            csv_dir = os.path.dirname(os.path.abspath(args.test_csv))
            alt_path = os.path.join(csv_dir, audio_path)
            if not os.path.exists(audio_path) and os.path.exists(alt_path):
                audio_path = alt_path

        try:
            # Generate prediction using the pre-loaded pipeline
            prediction = transcribe_audio(
                args.model_dir,
                audio_path,
                device=args.device,
                format="txt",
                asr_pipeline=asr_pipeline
            )

            # Normalize for fair evaluation
            norm_ref = normalize_amharic(reference)
            norm_pred = normalize_amharic(prediction)

            # Compute metrics for this sample
            # jiwer.wer and jiwer.cer can handle empty strings but we should be careful
            sample_wer = jiwer.wer(norm_ref, norm_pred) if norm_ref else (1.0 if norm_pred else 0.0)
            sample_cer = jiwer.cer(norm_ref, norm_pred) if norm_ref else (1.0 if norm_pred else 0.0)

            results.append({
                "audio_path": audio_path,
                "reference": reference,
                "prediction": prediction,
                "normalized_reference": norm_ref,
                "normalized_prediction": norm_pred,
                "wer": sample_wer,
                "cer": sample_cer
            })
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    if not results:
        print("No samples were successfully processed.")
        return

    results_df = pd.DataFrame(results)

    # Calculate overall metrics
    all_refs = results_df["normalized_reference"].tolist()
    all_preds = results_df["normalized_prediction"].tolist()

    overall_wer = jiwer.wer(all_refs, all_preds)
    overall_cer = jiwer.cer(all_refs, all_preds)

    print("\nEvaluation Summary:")
    print(f"Total samples: {len(results_df)}")
    print(f"Average WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    print(f"Average CER: {overall_cer:.4f} ({overall_cer*100:.2f}%)")

    results_df.to_csv(args.output_report, index=False)
    print(f"\nDetailed report saved to {args.output_report}")


if __name__ == "__main__":
    main()
