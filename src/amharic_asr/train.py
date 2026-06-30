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
    audio = batch[config.audio_column]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    # Normalize the transcript before tokenization
    normalized_text = normalize_amharic(batch[config.text_column])
    batch["labels"] = processor.tokenizer(normalized_text).input_ids
    return batch


def train_model(config: TrainingConfig) -> None:
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
    cer = evaluate.load("cer")

    def compute_metrics(eval_prediction):
        pred_ids = eval_prediction.predictions
        label_ids = eval_prediction.label_ids

        # Replace -100 with pad_token_id so they can be decoded
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Apply Amharic normalization for a fair evaluation during training.
        # This ensures that homophone differences don't penalize the model's score.
        pred_str = [normalize_amharic(p) for p in pred_str]
        label_str = [normalize_amharic(l) for l in label_str]

        # Filter out samples with empty reference to avoid issues with WER/CER calculation
        filtered_indices = [i for i, l in enumerate(label_str) if l.strip()]
        if not filtered_indices:
            return {"wer": 0.0, "cer": 0.0}

        filtered_preds = [pred_str[i] for i in filtered_indices]
        filtered_labels = [label_str[i] for i in filtered_indices]

        wer_val = 100 * wer.compute(predictions=filtered_preds, references=filtered_labels)
        cer_val = 100 * cer.compute(predictions=filtered_preds, references=filtered_labels)

        return {"wer": wer_val, "cer": cer_val}

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
