from typing import Any

import torch
from transformers import pipeline


def format_timestamp(seconds: float) -> str:
    """
    Converts seconds to SRT-compliant timestamp string (HH:MM:SS,mmm).
    """
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def to_srt(chunks: list[dict[str, Any]]) -> str:
    """
    Converts Whisper pipeline chunks into SRT format.
    """
    srt_lines = []
    for i, chunk in enumerate(chunks, 1):
        start, end = chunk["timestamp"]
        # Some chunks might not have an end timestamp if it's the very end
        if end is None:
            # Fallback or estimate based on start + some duration if needed
            # For now, we'll just use the start time to avoid crash if it's missing
            end = start + 1.0

        srt_lines.append(str(i))
        srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        srt_lines.append(chunk["text"].strip())
        srt_lines.append("")

    return "\n".join(srt_lines)


def transcribe_audio(
    model_dir: str,
    audio_path: str,
    device: int | None = None,
    chunk_length_s: int = 30,
    return_timestamps: bool = False,
) -> str:
    """
    Transcribes audio using a fine-tuned Whisper model.
    If return_timestamps is True, returns SRT formatted string.
    Otherwise, returns plain text.
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        device=device,
        chunk_length_s=chunk_length_s,
    )

    if return_timestamps:
        # return_timestamps="char" or True works for Whisper in pipeline
        result = asr(audio_path, return_timestamps=True)
        return to_srt(result["chunks"])
    else:
        result = asr(audio_path)
        return result["text"]
