import re
from datasets import Audio, DatasetDict, load_dataset

from .config import TrainingConfig


def normalize_amharic_text(text: str) -> str:
    """
    Normalizes Amharic text by standardizing homophones and removing punctuation.

    Amharic has several characters that sound the same but are written differently.
    Standardizing these helps the ASR model by reducing the vocabulary size and
    focusing on the phonetic content.

    Args:
        text: The raw Amharic text.

    Returns:
        The normalized Amharic text.
    """
    # Define homophone mappings for the 7 vowel orders
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

    # Remove Amharic (Ge'ez) and standard punctuation
    punctuation = "፣፤፥፦፧፨።,.:;!?\"'()[]{}«»"
    text = re.sub(f"[{re.escape(punctuation)}]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_amharic_dataset(config: TrainingConfig) -> DatasetDict:
    """
    Loads the Amharic dataset from CSV files and casts the audio column.

    Args:
        config: The training configuration containing file paths and settings.

    Returns:
        A DatasetDict containing the train and validation splits.
    """
    datasets = load_dataset(
        "csv",
        data_files={"train": config.train_csv, "validation": config.validation_csv},
    )
    datasets = datasets.cast_column(config.audio_column, Audio(sampling_rate=config.sampling_rate))
    return datasets
