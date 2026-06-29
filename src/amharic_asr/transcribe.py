from typing import Any

import torch
from transformers import pipeline


def format_timestamp(seconds: float, decimal_marker: str = ",") -> str:
    """Helper to convert seconds to timestamp format (HH:MM:SS,mmm or HH:MM:SS.mmm)."""
    td_hours = int(seconds // 3600)
    td_mins = int((seconds % 3600) // 60)
    td_secs = int(seconds % 60)
    td_millis = int(round((seconds % 1) * 1000))
    return f"{td_hours:02}:{td_mins:02}:{td_secs:02}{decimal_marker}{td_millis:03}"


def to_srt(chunks: list[dict[str, Any]]) -> str:
    """Converts pipeline output chunks with timestamps to SRT format."""
    srt_lines = []
    for i, chunk in enumerate(chunks, 1):
        start, end = chunk["timestamp"]
        if end is None:
            end = start
        start_str = format_timestamp(start, decimal_marker=",")
        end_str = format_timestamp(end, decimal_marker=",")
        text = chunk["text"].strip()
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_str} --> {end_str}")
        srt_lines.append(f"{text}\n")
    return "\n".join(srt_lines)


def to_vtt(chunks: list[dict[str, Any]]) -> str:
    """Converts pipeline output chunks with timestamps to WebVTT format."""
    vtt_lines = ["WEBVTT\n"]
    for chunk in chunks:
        start, end = chunk["timestamp"]
        if end is None:
            end = start
        start_str = format_timestamp(start, decimal_marker=".")
        end_str = format_timestamp(end, decimal_marker=".")
        text = chunk["text"].strip()
        vtt_lines.append(f"{start_str} --> {end_str}")
        vtt_lines.append(f"{text}\n")
    return "\n".join(vtt_lines)


def transcribe_audio(
    model_dir: str,
    audio_path: str,
    device: int | None = None,
    chunk_length_s: int = 30,
    format: str = "txt",
) -> str:
    """
    Generate transcript for an audio file.
    format can be 'txt', 'srt', or 'vtt'.
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    return_timestamps = format in ["srt", "vtt"]

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        device=device,
        chunk_length_s=chunk_length_s,
    )
    # Force Amharic language
    result = asr(
        audio_path,
        return_timestamps=return_timestamps,
        generate_kwargs={"language": "amharic"},
    )

    if format == "srt":
        return to_srt(result["chunks"])
    elif format == "vtt":
        return to_vtt(result["chunks"])

    return result["text"]
