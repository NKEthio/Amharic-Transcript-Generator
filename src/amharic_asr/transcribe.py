import torch
from transformers import pipeline


def transcribe_audio(
    model_dir: str,
    audio_path: str,
    device: int | None = None,
    chunk_length_s: int = 30,
) -> str:
    """
    Transcribes an audio file into Amharic text using a fine-tuned Whisper model.

    This function uses the Hugging Face 'pipeline' API for efficient inference,
    supporting long-form audio through chunking.

    Args:
        model_dir: Path to the directory containing the fine-tuned model and processor.
        audio_path: Path to the audio file to be transcribed.
        device: Device to run inference on (0 for GPU, -1 for CPU). If None, it auto-detects.
        chunk_length_s: Length of audio chunks in seconds for long-form transcription.

    Returns:
        The generated Amharic transcript as a string.
    """
    # Auto-detect CUDA if device is not specified
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    # Initialize the ASR pipeline with the fine-tuned model
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        device=device,
        chunk_length_s=chunk_length_s,
    )

    # Run the audio through the pipeline and return the text
    result = asr(audio_path)
    return result["text"]
