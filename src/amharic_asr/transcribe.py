import torch
from transformers import pipeline


def transcribe_audio(
    model_dir: str,
    audio_path: str,
    device: int | None = None,
    chunk_length_s: int = 30,
) -> str:
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        device=device,
        chunk_length_s=chunk_length_s,
    )
    result = asr(audio_path)
    return result["text"]
