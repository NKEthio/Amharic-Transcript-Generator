import argparse
import os
import re
from dataclasses import dataclass, field
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
    """Configuration for Amharic Whisper fine-tuning."""
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
    # Automatically use fp16 if CUDA is available
    fp16: bool = field(default_factory=lambda: torch.cuda.is_available())
    preprocessing_num_proc: int = 1


def load_training_config(path: str) -> TrainingConfig:
    """Loads training configuration from YAML."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TrainingConfig(**raw)


def normalize_amharic_text(text: str) -> str:
    """
    Standardizes Amharic homophones and removes punctuation.
    """
    # Group 1: ሀ, ሐ, ኀ -> ሀ
    text = re.sub("[ሐኀ]", "ሀ", text)
    text = re.sub("[ሑኁ]", "ሁ", text)
    text = re.sub("[ሒኂ]", "ሂ", text)
    text = re.sub("[ሓኃ]", "ሃ", text)
    text = re.sub("[ሔኄ]", "ሄ", text)
    text = re.sub("[ሕኅ]", "ህ", text)
    text = re.sub("[ሖኆ]", "ሆ", text)

    # Group 2: ሰ, ሠ -> ሰ
    text = re.sub("ሠ", "ሰ", text)
    text = re.sub("ሡ", "ሱ", text)
    text = re.sub("ሢ", "ሲ", text)
    text = re.sub("ሣ", "ሳ", text)
    text = re.sub("ሤ", "ሴ", text)
    text = re.sub("ሥ", "ስ", text)
    text = re.sub("ሦ", "ሶ", text)

    # Group 3: አ, ዐ -> አ
    text = re.sub("ዐ", "አ", text)
    text = re.sub("ዑ", "ኡ", text)
    text = re.sub("ዒ", "ኢ", text)
    text = re.sub("ዓ", "ኣ", text)
    text = re.sub("ዔ", "ኤ", text)
    text = re.sub("ዕ", "እ", text)
    text = re.sub("ዖ", "ኦ", text)

    # Group 4: ጸ, ፀ -> ጸ
    text = re.sub("ፀ", "ጸ", text)
    text = re.sub("ፁ", "ጹ", text)
    text = re.sub("ፂ", "ጺ", text)
    text = re.sub("ፃ", "ጻ", text)
    text = re.sub("ፄ", "ጼ", text)
    text = re.sub("ፅ", "ጽ", text)
    text = re.sub("ፆ", "ጾ", text)

    # Remove Amharic and standard punctuation
    punctuation = "፣፤፥፦፧፨።,.:;!?\"'()[]{}«»"
    text = re.sub(f"[{re.escape(punctuation)}]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_amharic_dataset(config: TrainingConfig) -> DatasetDict:
    """Loads CSV dataset and casts audio column."""
    datasets = load_dataset(
        "csv",
        data_files={"train": config.train_csv, "validation": config.validation_csv},
    )
    datasets = datasets.cast_column(config.audio_column, Audio(sampling_rate=config.sampling_rate))
    return datasets


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Collator for Whisper fine-tuning."""
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        bos_token = self.processor.tokenizer.bos_token_id
        if bos_token is not None and labels.numel() > 0 and labels.size(1) > 0:
            if (labels[:, 0] == bos_token).all().item():
                labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def _prepare_dataset(batch: dict[str, Any], processor: WhisperProcessor, config: TrainingConfig) -> dict[str, Any]:
    """Preprocesses a dataset batch: feature extraction and transcript normalization."""
    audio = batch[config.audio_column]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    # Apply Amharic text normalization
    normalized_text = normalize_amharic_text(batch[config.text_column])
    batch["labels"] = processor.tokenizer(normalized_text).input_ids
    return batch


def train_model(config: TrainingConfig) -> None:
    """Main training function."""
    processor = WhisperProcessor.from_pretrained(config.base_model, language="am", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(config.base_model)
    model.generation_config.language = "am"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

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
        pred_ids = eval_prediction.predictions
        label_ids = eval_prediction.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer.compute(predictions=pred_str, references=label_str)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_train_epochs,
        fp16=config.fp16,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)


def transcribe_audio(
    model_dir: str,
    audio_path: str,
    device: int | None = None,
    chunk_length_s: int = 30,
) -> str:
    """Inference function for transcribing audio."""
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        device=device,
        chunk_length_s=chunk_length_s,
    )
    result = asr(audio_path)
    return result["text"]


def main() -> None:
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
        )
        print(text)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
