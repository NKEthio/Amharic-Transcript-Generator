from typing import Any

import torch
from transformers import pipeline


def format_timestamp(seconds: float) -> str:
    """Helper to convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    td_hours = int(seconds // 3600)
    td_mins = int((seconds % 3600) // 60)
    td_secs = int(seconds % 60)
    td_millis = int(round((seconds % 1) * 1000))
    return f"{td_hours:02}:{td_mins:02}:{td_secs:02},{td_millis:03}"


def to_srt(chunks: list[dict[str, Any]]) -> str:
    """Converts pipeline output chunks with timestamps to SRT format."""
    srt_lines = []
    for i, chunk in enumerate(chunks, 1):
        start, end = chunk["timestamp"]
        # Some chunks might have None as end if it's the last one
        if end is None:
            # We don't really know the end, but we can't leave it None for SRT
            # As a fallback, we could use the start + some duration or just the same as start
            end = start
        start_str = format_timestamp(start)
        end_str = format_timestamp(end)
        text = chunk["text"].strip()
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_str} --> {end_str}")
        srt_lines.append(f"{text}\n")
    return "\n".join(srt_lines)


def transcribe_audio(
    model_dir: str,
    audio_path: str,
    device: int | None = None,
    chunk_length_s: int = 30,
    return_timestamps: bool = False,
) -> str:
    """
    Generate transcript for an audio file.
    If return_timestamps is True, returns SRT formatted string.
    Otherwise returns plain text.
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        device=device,
        chunk_length_s=chunk_length_s,
    )
    result = asr(audio_path, return_timestamps=return_timestamps)

    if return_timestamps:
        return to_srt(result["chunks"])

    return result["text"]
