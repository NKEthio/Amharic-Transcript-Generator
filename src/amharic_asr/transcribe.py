import torch
from transformers import pipeline
from typing import Any


def format_timestamp(seconds: float, vtt: bool = False) -> str:
    """
    Formats seconds into HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT).
    """
    msec = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    separator = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{msec:03d}"


def to_srt(chunks: list[dict[str, Any]]) -> str:
    """
    Converts pipeline chunks with timestamps to SRT format.
    """
    srt_lines = []
    for i, chunk in enumerate(chunks, 1):
        start, end = chunk["timestamp"]
        # Some chunks might not have an end timestamp if it's the very end
        if end is None:
            end = start + 1.0  # Fallback

        srt_lines.append(str(i))
        srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        srt_lines.append(chunk["text"].strip())
        srt_lines.append("")
    return "\n".join(srt_lines)


def to_vtt(chunks: list[dict[str, Any]]) -> str:
    """
    Converts pipeline chunks with timestamps to VTT format.
    """
    vtt_lines = ["WEBVTT", ""]
    for chunk in chunks:
        start, end = chunk["timestamp"]
        if end is None:
            end = start + 1.0

        vtt_lines.append(f"{format_timestamp(start, vtt=True)} --> {format_timestamp(end, vtt=True)}")
        vtt_lines.append(chunk["text"].strip())
        vtt_lines.append("")
    return "\n".join(vtt_lines)


def transcribe_audio(
    model_dir: str,
    audio_path: str,
    device: int | None = None,
    chunk_length_s: int = 30,
    return_timestamps: bool | str = False,
) -> dict[str, Any]:
    """
    Transcribes audio using a fine-tuned Whisper model.
    Returns the full pipeline output dictionary.
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        device=device,
        chunk_length_s=chunk_length_s,
    )
    # return_timestamps can be True, False, or "word"
    result = asr(audio_path, return_timestamps=return_timestamps)
    return result
