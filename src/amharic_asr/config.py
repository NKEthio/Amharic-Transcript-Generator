from dataclasses import dataclass

import yaml


@dataclass
class TrainingConfig:
    """
    Configuration dataclass for Amharic ASR fine-tuning.

    Attributes:
        base_model: Identifier for the pre-trained model (e.g., 'openai/whisper-small').
        train_csv: Path to the training dataset CSV.
        validation_csv: Path to the validation dataset CSV.
        audio_column: Name of the column containing audio paths.
        text_column: Name of the column containing transcripts.
        sampling_rate: Target audio sampling rate (Whisper requires 16000).
        output_dir: Directory to save model checkpoints and logs.
        num_train_epochs: Number of full passes through the training data.
        learning_rate: Initial learning rate for the optimizer.
        warmup_steps: Number of steps to linearly increase the learning rate.
        per_device_train_batch_size: Batch size per GPU for training.
        per_device_eval_batch_size: Batch size per GPU for evaluation.
        gradient_accumulation_steps: Number of update steps to accumulate before a backward pass.
        logging_steps: Frequency of logging training progress.
        save_steps: Frequency of saving model checkpoints.
        eval_steps: Frequency of evaluating the model.
        fp16: Whether to use half-precision (16-bit) floating point for training.
        preprocessing_num_proc: Number of CPU processes for data preparation.
    """
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
    """
    Loads training configuration from a YAML file and returns a TrainingConfig object.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        An instance of TrainingConfig populated with values from the YAML file.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TrainingConfig(**raw)
