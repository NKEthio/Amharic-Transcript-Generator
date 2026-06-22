from dataclasses import dataclass
from typing import Any

import evaluate
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from .config import TrainingConfig
from .data import load_amharic_dataset, normalize_amharic


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs and labels to the maximum length
    in the batch. This ensures efficient training by avoiding static padding.
    """
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Extract audio features and pad them to the batch max length
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Extract labels (tokenized text) and pad them
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Use -100 to mask padding tokens in labels so they are ignored during loss calculation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove the BOS token if it's already there, as it will be added by the model during training
        bos_token = self.processor.tokenizer.bos_token_id
        if bos_token is not None and labels.numel() > 0 and labels.size(1) > 0:
            if (labels[:, 0] == bos_token).all().item():
                labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def _prepare_dataset(batch: dict[str, Any], processor: WhisperProcessor, config: TrainingConfig) -> dict[str, Any]:
    """
    Preprocesses a batch of raw data by:
    1. Extracting log-Mel features from audio.
    2. Normalizing the Amharic transcript.
    3. Tokenizing the normalized transcript.

    Args:
        batch: A single example or batch from the dataset.
        processor: The Whisper processor (feature extractor + tokenizer).
        config: TrainingConfig object.

    Returns:
        The processed batch with 'input_features' and 'labels'.
    """
    audio = batch[config.audio_column]

    # Preprocess audio into Mel spectrograms
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    # Normalize Amharic text to handle homophones and remove punctuation
    normalized_text = normalize_amharic(batch[config.text_column])

    # Convert normalized text into token IDs
    batch["labels"] = processor.tokenizer(normalized_text).input_ids
    return batch


def train_model(config: TrainingConfig) -> None:
    """
    Orchestrates the fine-tuning process for the Whisper model on Amharic data.

    Steps:
    1. Initialize the model and processor.
    2. Load and preprocess the dataset.
    3. Define training arguments and the trainer.
    4. Start training and save the final artifacts.

    Args:
        config: TrainingConfig object.
    """
    # Initialize processor and model with Amharic language settings
    processor = WhisperProcessor.from_pretrained(config.base_model, language="am", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(config.base_model)

    # Configure generation parameters to force Amharic transcription
    model.generation_config.language = "am"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Load and map preprocessing across the entire dataset
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
        """Calculates the Word Error Rate (WER) during evaluation."""
        pred_ids = eval_prediction.predictions
        label_ids = eval_prediction.label_ids

        # Replace -100 mask with pad_token_id for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode token IDs back to strings
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Calculate WER percentage
        return {"wer": 100 * wer.compute(predictions=pred_str, references=label_str)}

    # Define training parameters
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

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # Execute fine-tuning
    trainer.train()

    # Save both model weights and processor config for future inference
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
