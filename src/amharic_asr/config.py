from dataclasses import dataclass

import yaml


@dataclass
class TrainingConfig:
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
    fp16: bool = False
    preprocessing_num_proc: int = 1


def load_training_config(path: str) -> TrainingConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TrainingConfig(**raw)
