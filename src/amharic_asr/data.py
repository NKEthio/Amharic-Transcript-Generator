from datasets import Audio, DatasetDict, load_dataset

from .config import TrainingConfig


def load_amharic_dataset(config: TrainingConfig) -> DatasetDict:
    datasets = load_dataset(
        "csv",
        data_files={"train": config.train_csv, "validation": config.validation_csv},
    )
    datasets = datasets.cast_column(config.audio_column, Audio(sampling_rate=config.sampling_rate))
    return datasets
