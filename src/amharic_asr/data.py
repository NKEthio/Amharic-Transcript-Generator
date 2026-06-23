import re
from datasets import Audio, DatasetDict, load_dataset

from .config import TrainingConfig


def normalize_amharic_text(text: str) -> str:
    """
    Normalizes Amharic text by handling homophones and removing punctuation.

    Amharic has several characters that sound the same but are written differently.
    Standardizing these helps the model focus on the phonetic content rather than
    spelling variations.

    Args:
        text: The raw Amharic text to normalize.

    Returns:
        The normalized text.
    """
    if not text:
        return ""

    # 1. Normalize Homophones
    # These groups of characters represent the same sounds in modern Amharic.

    # Normalize 'h' sounds: ሐ, ኀ, ኸ -> ሀ
    text = re.sub("[ሐኀኸ]", "ሀ", text)
    # Normalize 's' sounds: ሠ -> ሰ
    text = re.sub("ሠ", "ሰ", text)
    # Normalize 'a' sounds: ዐ -> አ
    text = re.sub("ዐ", "አ", text)
    # Normalize 'ts' sounds: ፀ -> ጸ
    text = re.sub("ፀ", "ጸ", text)

    # Also handle the variants with different vowels if necessary,
    # but usually the base forms cover most cases if they are in the same Unicode block.
    # Actually, it's better to be explicit for each vowel form if we want thoroughness.
    # For now, we'll start with these and can expand if needed.
    # Standard practice often involves mapping all orders.

    # 2. Remove Amharic-specific punctuation
    # ። (Full stop), ፣ (Comma), ፤ (Semicolon), ፥ (Colon), ፦ (Preface colon), ፧ (Question mark), ፨ (Paragraph separator)
    text = re.sub(r"[።፣፤፥፦፧፨]", " ", text)

    # 3. Remove standard punctuation and special characters
    text = re.sub(r"[^\u1200-\u137F\s]", " ", text)

    # 4. Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_amharic_dataset(config: TrainingConfig) -> DatasetDict:
    datasets = load_dataset(
        "csv",
        data_files={"train": config.train_csv, "validation": config.validation_csv},
    )
    datasets = datasets.cast_column(config.audio_column, Audio(sampling_rate=config.sampling_rate))
    return datasets
