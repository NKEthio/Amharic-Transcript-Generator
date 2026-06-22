import re
from datasets import Audio, DatasetDict, load_dataset

from .config import TrainingConfig


def normalize_amharic(text: str) -> str:
    """
    Normalizes Amharic text by handling homophones and removing punctuation.

    Amharic has several characters that sound the same but are written differently.
    Mapping them to a single standard form helps the model learn more efficiently
    by reducing vocabulary redundancy.

    Args:
        text: Input Amharic string.

    Returns:
        Normalized Amharic string.
    """
    # 'ha' family normalization: ሐ and ኀ families are mapped to the ሀ family
    text = text.replace("ሐ", "ሀ").replace("ሑ", "ሁ").replace("ሒ", "ሂ").replace("ሓ", "ሃ").replace("ሔ", "ሄ").replace("ሕ", "ህ").replace("ሖ", "ሆ")
    text = text.replace("ኀ", "ሀ").replace("ኁ", "ሁ").replace("ኂ", "ሂ").replace("ኃ", "ሃ").replace("ኄ", "ሄ").replace("ኅ", "ህ").replace("ኆ", "ሆ")

    # 'se' family normalization: ሠ family is mapped to the ሰ family
    text = text.replace("ሠ", "ሰ").replace("ሡ", "ሱ").replace("ሢ", "ሲ").replace("ሣ", "ሳ").replace("ሤ", "ሴ").replace("ሥ", "ስ").replace("ሦ", "ሶ")

    # 'a' family normalization: ዐ family is mapped to the አ family
    text = text.replace("ዐ", "አ").replace("ዑ", "ኡ").replace("ዒ", "ኢ").replace("ዓ", "ኣ").replace("ዔ", "ኤ").replace("ዕ", "እ").replace("ዖ", "ኦ")

    # 'tse' family normalization: ፀ family is mapped to the ጸ family
    text = text.replace("ፀ", "ጸ").replace("ፁ", "ጹ").replace("ፂ", "ጺ").replace("ፃ", "ጻ").replace("ፄ", "ጼ").replace("ፅ", "ጽ").replace("ፆ", "ጾ")

    # Remove Ge'ez specific punctuation marks
    text = re.sub(r'[፡።፣፤፥፦፧፨]', '', text)

    # Remove standard Western punctuation marks
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', '', text)

    # Clean up extra spaces and trim the string
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_amharic_dataset(config: TrainingConfig) -> DatasetDict:
    """
    Loads the Amharic dataset from CSV files specified in the configuration.

    The dataset is expected to have an audio path column and a transcript column.
    The audio files are automatically resampled to the rate required by the model.

    Args:
        config: TrainingConfig object containing dataset paths and sampling rate.

    Returns:
        A DatasetDict containing 'train' and 'validation' splits.
    """
    # Load the raw CSV data into a Hugging Face DatasetDict
    datasets = load_dataset(
        "csv",
        data_files={"train": config.train_csv, "validation": config.validation_csv},
    )

    # Ensure the audio column is treated as audio data and resampled to 16kHz
    datasets = datasets.cast_column(config.audio_column, Audio(sampling_rate=config.sampling_rate))

    return datasets
