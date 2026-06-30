from typing import Any

import torch
import librosa
from transformers import pipeline


def format_timestamp(seconds: float, format: str = "srt") -> str:
    """
    Helper to convert seconds to timestamp format.
    SRT: HH:MM:SS,mmm
    VTT: HH:MM:SS.mmm
    """
    td_hours = int(seconds // 3600)
    td_mins = int((seconds % 3600) // 60)
    td_secs = int(seconds % 60)
    td_millis = int(round((seconds % 1) * 1000))

    separator = "," if format == "srt" else "."
    return f"{td_hours:02}:{td_mins:02}:{td_secs:02}{separator}{td_millis:03}"


def to_srt(chunks: list[dict[str, Any]]) -> str:
    """Converts pipeline output chunks with timestamps to SRT format."""
    srt_lines = []
    for i, chunk in enumerate(chunks, 1):
        start, end = chunk["timestamp"]
        # Some chunks might have None as end if it's the last one
        if end is None:
            end = start
        start_str = format_timestamp(start, format="srt")
        end_str = format_timestamp(end, format="srt")
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
        start_str = format_timestamp(start, format="vtt")
        end_str = format_timestamp(end, format="vtt")
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
    Generate transcript for an audio file in the specified format (txt, srt, vtt).
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    # Force Amharic language for transcription
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        device=device,
        chunk_length_s=chunk_length_s,
    )

    audio, sr = librosa.load(audio_path, sr=16000)

    return_timestamps = (format in ["srt", "vtt"])
    result = asr(
        audio,
        return_timestamps=return_timestamps,
        generate_kwargs={"language": "amharic"}
    )

    if format == "srt":
        return to_srt(result["chunks"])
    elif format == "vtt":
        return to_vtt(result["chunks"])

    return result["text"]
