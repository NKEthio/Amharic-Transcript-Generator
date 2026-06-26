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
from .data import load_amharic_dataset, normalize_amharic_text


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received for Whisper fine-tuning.
    """
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Split inputs and labels since they have different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features and labels
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly during training
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous steps, remove it as it's added by the model
        bos_token = self.processor.tokenizer.bos_token_id
        if bos_token is not None and labels.numel() > 0 and labels.size(1) > 0:
            if (labels[:, 0] == bos_token).all().item():
                labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def _prepare_dataset(batch: dict[str, Any], processor: WhisperProcessor, config: TrainingConfig) -> dict[str, Any]:
    """
    Preprocesses a single batch of the dataset.

    Includes audio feature extraction and transcript normalization + tokenization.
    """
    # Load and resample audio
    audio = batch[config.audio_column]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    # Normalize Amharic transcript before tokenization
    normalized_text = normalize_amharic_text(batch[config.text_column])
    batch["labels"] = processor.tokenizer(normalized_text).input_ids
    return batch


def train_model(config: TrainingConfig) -> None:
    """
    Main training loop for fine-tuning Whisper on Amharic data.
    """
    # Initialize processor and model
    processor = WhisperProcessor.from_pretrained(config.base_model, language="am", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(config.base_model)

    # Configure generation settings for Amharic
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
        """
        Calculates Word Error Rate (WER) for evaluation.
        """
        pred_ids = eval_prediction.predictions
        label_ids = eval_prediction.label_ids

        # Replace -100 with pad_token_id to decode correctly
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Calculate WER percentage
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

    # Start fine-tuning
    trainer.train()

    # Save the final model and processor
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
