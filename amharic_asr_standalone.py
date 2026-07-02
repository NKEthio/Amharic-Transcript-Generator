import argparse
import os
import re
from dataclasses import dataclass
from typing import Any

import evaluate
import torch
import yaml
from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning the Amharic ASR model."""
    base_model: str
    train_csv: str
    validation_csv: str
    audio_column: str = "audio_path"
    text_column: str = "transcript"
    sampling_rate: int = 16000
    output_dir: str = "outputs/amharic-whisper-small-ft"
    num_train_epochs: int = 6
    learning_rate: float = 1e-5
    warmup_steps: int = 200
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    logging_steps: int = 25
    save_steps: int = 250
    eval_steps: int = 250
    max_steps: int = -1
    generation_max_length: int = 225
    save_total_limit: int = 3
    fp16: bool = torch.cuda.is_available()
    preprocessing_num_proc: int = 1


def normalize_amharic(text: str) -> str:
    """
    Normalizes Amharic text by handling homophones and removing punctuation.
    This ensures consistency during training and evaluation.
    """
    if not text:
        return ""

    # Remove Amharic punctuation (Ge'ez punctuation marks)
    text = re.sub(r"[\u1361-\u1368]", " ", text)
    # Remove standard punctuation
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', " ", text)

    # Normalize homophones to a single standard form
    # ሀ, ሐ, ኀ -> ሀ
    text = re.sub(r"[ሐኀ]", "ሀ", text)
    text = re.sub(r"[ሑኁ]", "ሁ", text)
    text = re.sub(r"[ሒኂ]", "ሂ", text)
    text = re.sub(r"[ሓኃ]", "ሃ", text)
    text = re.sub(r"[ሔኄ]", "ሄ", text)
    text = re.sub(r"[ሕኅ]", "ህ", text)
    text = re.sub(r"[ሖኆ]", "ሆ", text)

    # ሰ, ሠ -> ሰ
    text = re.sub(r"ሠ", "ሰ", text)
    text = re.sub(r"ሡ", "ሱ", text)
    text = re.sub(r"ሢ", "ሲ", text)
    text = re.sub(r"ሣ", "ሳ", text)
    text = re.sub(r"ሤ", "ሴ", text)
    text = re.sub(r"ሥ", "ስ", text)
    text = re.sub(r"ሦ", "ሶ", text)

    # አ, ዐ -> አ
    text = re.sub(r"ዐ", "አ", text)
    text = re.sub(r"ዑ", "ኡ", text)
    text = re.sub(r"ዒ", "ኢ", text)
    text = re.sub(r"ዓ", "ኣ", text)
    text = re.sub(r"ዔ", "ኤ", text)
    text = re.sub(r"ዕ", "እ", text)
    text = re.sub(r"ዖ", "ኦ", text)

    # ጸ, ፀ -> ጸ
    text = re.sub(r"ፀ", "ጸ", text)
    text = re.sub(r"ፁ", "ጹ", text)
    text = re.sub(r"ፂ", "ጺ", text)
    text = re.sub(r"ፃ", "ጻ", text)
    text = re.sub(r"ፄ", "ጼ", text)
    text = re.sub(r"ፅ", "ጽ", text)
    text = re.sub(r"ፆ", "ጾ", text)

    # Remove extra whitespace and trim
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_training_config(path: str) -> TrainingConfig:
    """Loads training configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TrainingConfig(**raw)


def load_amharic_dataset(config: TrainingConfig) -> DatasetDict:
    """Loads the Amharic dataset from CSV files and casts the audio column."""
    datasets = load_dataset(
        "csv",
        data_files={"train": config.train_csv, "validation": config.validation_csv},
    )
    datasets = datasets.cast_column(config.audio_column, Audio(sampling_rate=config.sampling_rate))
    return datasets


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator that handles dynamic padding for ASR features and labels."""
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 so it's ignored by the loss function
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove beginning-of-sentence token if present to let the model add its own
        bos_token = self.processor.tokenizer.bos_token_id
        if bos_token is not None and labels.numel() > 0 and labels.size(1) > 0:
            if (labels[:, 0] == bos_token).all().item():
                labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def _prepare_dataset(batch: dict[str, Any], processor: WhisperProcessor, config: TrainingConfig) -> dict[str, Any]:
    """Helper function to preprocess individual dataset batches."""
    audio = batch[config.audio_column]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    # Normalize the transcript before tokenization for consistency
    normalized_text = normalize_amharic(batch[config.text_column])
    batch["labels"] = processor.tokenizer(normalized_text).input_ids
    return batch


def train_model(config: TrainingConfig) -> None:
    """Fine-tunes the Whisper model on the Amharic dataset."""
    # Load processor and model
    processor = WhisperProcessor.from_pretrained(config.base_model, language="am", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(config.base_model)

    # Configure generation parameters
    model.generation_config.language = "am"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Load and preprocess dataset
    datasets = load_amharic_dataset(config)
    dataset_columns = datasets["train"].column_names
    datasets = datasets.map(
        _prepare_dataset,
        fn_kwargs={"processor": processor, "config": config},
        remove_columns=dataset_columns,
        num_proc=config.preprocessing_num_proc,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer = evaluate.load("wer")

    def compute_metrics(eval_prediction):
        """Calculates WER for evaluation."""
        pred_ids = eval_prediction.predictions
        label_ids = eval_prediction.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer.compute(predictions=pred_str, references=label_str)}

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        fp16=config.fp16,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        predict_with_generate=True,
        generation_max_length=config.generation_max_length,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=[],
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # Start training and save artifacts
    trainer.train()
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)


def format_timestamp(seconds: float, format: str = "srt") -> str:
    """Helper to convert seconds to timestamp format (SRT or VTT)."""
    td_hours = int(seconds // 3600)
    td_mins = int((seconds % 3600) // 60)
    td_secs = int(seconds % 60)
    td_millis = int(round((seconds % 1) * 1000))
    separator = "," if format == "srt" else "."
    return f"{td_hours:02}:{td_mins:02}:{td_secs:02}{separator}{td_millis:03}"


def to_srt(chunks: list[dict[str, Any]]) -> str:
    """Converts pipeline output chunks with timestamps to SRT format."""
    srt_lines = []
    for i, chunk in enumerate(chunks, 1):
        start, end = chunk["timestamp"]
        if end is None:
            end = start
        start_str = format_timestamp(start, format="srt")
        end_str = format_timestamp(end, format="srt")
        text = chunk["text"].strip()
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_str} --> {end_str}")
        srt_lines.append(f"{text}\n")
    return "\n".join(srt_lines)


def to_vtt(chunks: list[dict[str, Any]]) -> str:
    """Converts pipeline output chunks with timestamps to WebVTT format."""
    vtt_lines = ["WEBVTT\n"]
    for chunk in chunks:
        start, end = chunk["timestamp"]
        if end is None:
            end = start
        start_str = format_timestamp(start, format="vtt")
        end_str = format_timestamp(end, format="vtt")
        text = chunk["text"].strip()
        vtt_lines.append(f"{start_str} --> {end_str}")
        vtt_lines.append(f"{text}\n")
    return "\n".join(vtt_lines)


def load_transcription_pipeline(
    model_dir: str,
    device: int | None = None,
    chunk_length_s: int = 30,
) -> Any:
    """Initialize the ASR pipeline for inference."""
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        device=device,
        chunk_length_s=chunk_length_s,
    )


def transcribe_audio(
    model_dir: str,
    audio_path: str,
    device: int | None = None,
    chunk_length_s: int = 30,
    format: str = "txt",
    asr_pipeline: Any = None,
) -> str:
    """Generate transcript for an audio file, optionally reusing an existing pipeline."""
    if asr_pipeline is not None:
        asr = asr_pipeline
    else:
        asr = load_transcription_pipeline(
            model_dir=model_dir,
            device=device,
            chunk_length_s=chunk_length_s
        )

    return_timestamps = (format in ["srt", "vtt"])
    result = asr(
        audio_path,
        return_timestamps=return_timestamps,
        generate_kwargs={"language": "amharic"}
    )

    if format == "srt":
        return to_srt(result["chunks"])
    elif format == "vtt":
        return to_vtt(result["chunks"])

    return result["text"]


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Amharic ASR Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train subparser
    train_parser = subparsers.add_parser("train", help="Fine-tune the model")
    train_parser.add_argument("--config", required=True, help="Path to training YAML config.")

    # Transcribe subparser
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio")
    transcribe_parser.add_argument("--model-dir", required=True, help="Path to model directory.")
    transcribe_parser.add_argument("--audio-path", required=True, help="Path to audio file.")
    transcribe_parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Device index, use -1 for CPU.",
    )
    transcribe_parser.add_argument(
        "--chunk-length-s",
        type=int,
        default=30,
        help="Chunk size in seconds.",
    )
    transcribe_parser.add_argument(
        "--format",
        choices=["txt", "srt", "vtt"],
        default="txt",
        help="Output format (txt, srt, or vtt).",
    )

    args = parser.parse_args()

    if args.command == "train":
        config = load_training_config(args.config)
        train_model(config)
    elif args.command == "transcribe":
        text = transcribe_audio(
            args.model_dir,
            args.audio_path,
            device=args.device,
            chunk_length_s=args.chunk_length_s,
            format=args.format,
        )
        print(text)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
