import re

from datasets import Audio, DatasetDict, load_dataset

from .config import TrainingConfig


def normalize_amharic(text: str) -> str:
    """
    Normalizes Amharic text by handling homophones and removing punctuation.
    """
    if not text:
        return ""

    # Remove Amharic punctuation
    text = re.sub(r"[\u1361-\u1368]", " ", text)
    # Remove standard punctuation
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', " ", text)

    # Normalize homophones
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

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_amharic_dataset(config: TrainingConfig) -> DatasetDict:
    datasets = load_dataset(
        "csv",
        data_files={"train": config.train_csv, "validation": config.validation_csv},
    )
    datasets = datasets.cast_column(config.audio_column, Audio(sampling_rate=config.sampling_rate))
    return datasets
