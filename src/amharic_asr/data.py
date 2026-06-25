import re
from datasets import Audio, DatasetDict, load_dataset

from .config import TrainingConfig


def normalize_amharic_text(text: str) -> str:
    """
    Standardize Amharic text by normalizing homophones and removing punctuation.
    This helps in reducing the vocabulary size and improves ASR performance by
    treating different spellings of the same sound as the same token.

    Homophones normalized:
    - ሀ, ሐ, ኀ -> ሀ (and all their vowel orders)
    - ሰ, ሠ -> ሰ (and all their vowel orders)
    - አ, ዐ -> አ (and all their vowel orders)
    - ጸ, ፀ -> ጸ (and all their vowel orders)
    """
    # Homophone mappings for all 7 orders
    # Order 1: ha, sa, a, tsa
    text = re.sub("[ሐኀ]", "ሀ", text)
    text = re.sub("ሠ", "ሰ", text)
    text = re.sub("ዐ", "አ", text)
    text = re.sub("ፀ", "ጸ", text)

    # Order 2: hu, su, u, tsu
    text = re.sub("[ሑኁ]", "ሁ", text)
    text = re.sub("ሡ", "ሱ", text)
    text = re.sub("ዑ", "ኡ", text)
    text = re.sub("ፁ", "ጹ", text)

    # Order 3: hi, si, i, tsi
    text = re.sub("[ሒኂ]", "ሂ", text)
    text = re.sub("ሢ", "ሲ", text)
    text = re.sub("ዒ", "ኢ", text)
    text = re.sub("ፂ", "ጺ", text)

    # Order 4: ha, sa, a, tsa
    text = re.sub("[ሓኃ]", "ሃ", text)
    text = re.sub("ሣ", "ሳ", text)
    text = re.sub("ዓ", "ኣ", text)
    text = re.sub("ፃ", "ጻ", text)

    # Order 5: he, se, e, tse
    text = re.sub("[ሔኄ]", "ሄ", text)
    text = re.sub("ሤ", "ሴ", text)
    text = re.sub("ዔ", "ኤ", text)
    text = re.sub("ፄ", "ጼ", text)

    # Order 6: h, s, e, ts
    text = re.sub("[ሕኅ]", "ህ", text)
    text = re.sub("ሥ", "ስ", text)
    text = re.sub("ዕ", "እ", text)
    text = re.sub("ፅ", "ጽ", text)

    # Order 7: ho, so, o, tso
    text = re.sub("[ሖኆ]", "ሆ", text)
    text = re.sub("ሦ", "ሶ", text)
    text = re.sub("ዖ", "ኦ", text)
    text = re.sub("ፆ", "ጾ", text)

    # Remove Ge'ez punctuation
    # ፡ (space), ። (full stop), ፣ (comma), ፤ (semicolon), ፥ (colon), ፦ (preface colon), ፧ (question mark), ፨ (paragraph separator)
    text = re.sub("[፡።፣፤፥፦፧፨]", " ", text)

    # Remove standard punctuation and extra whitespace
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_amharic_dataset(config: TrainingConfig) -> DatasetDict:
    datasets = load_dataset(
        "csv",
        data_files={"train": config.train_csv, "validation": config.validation_csv},
    )
    datasets = datasets.cast_column(config.audio_column, Audio(sampling_rate=config.sampling_rate))
    return datasets
