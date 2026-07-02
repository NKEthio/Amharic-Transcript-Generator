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


def load_transcription_pipeline(
    model_dir: str,
    device: int | None = None,
    chunk_length_s: int = 30,
) -> Any:
    """
    Load the automatic-speech-recognition pipeline for Amharic.

    Args:
        model_dir: Path to the fine-tuned model directory.
        device: Device index (e.g., 0 for GPU, -1 for CPU). If None, defaults to GPU if available.
        chunk_length_s: The maximum length of each audio chunk in seconds.

    Returns:
        The initialized Hugging Face pipeline.
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    # Initialize the ASR pipeline with the specified model and settings
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        device=device,
        chunk_length_s=chunk_length_s,
    )
    return asr


def transcribe_audio(
    model_dir: str,
    audio_path: str,
    device: int | None = None,
    chunk_length_s: int = 30,
    format: str = "txt",
    task: str = "transcribe",
    asr_pipeline: Any = None,
) -> str:
    """
    Generate transcript for an audio file in the specified format (txt, srt, vtt).
    The 'task' can be either 'transcribe' (default) or 'translate' (to English).

    Args:
        model_dir: Path to the fine-tuned model directory. (Ignored if asr_pipeline is provided)
        audio_path: Path to the audio file to transcribe.
        device: Device index. (Ignored if asr_pipeline is provided)
        chunk_length_s: Chunk length in seconds. (Ignored if asr_pipeline is provided)
        format: Desired output format ('txt', 'srt', 'vtt').
        task: Task to perform ('transcribe' or 'translate').
        asr_pipeline: An optional pre-loaded ASR pipeline to reuse.

    Returns:
        The generated transcript as a string.
    """
    # Use the provided pipeline or load a new one if not available
    if asr_pipeline is not None:
        asr = asr_pipeline
    else:
        asr = load_transcription_pipeline(
            model_dir=model_dir,
            device=device,
            chunk_length_s=chunk_length_s
        )

    # Load audio file and resample to 16kHz as required by Whisper
    audio, sr = librosa.load(audio_path, sr=16000)

    # Determine if timestamps are needed based on the output format
    return_timestamps = (format in ["srt", "vtt"])

    # Execute the ASR task
    result = asr(
        audio,
        return_timestamps=return_timestamps,
        generate_kwargs={"language": "amharic", "task": task}
    )

    # Format the output according to the requested format
    if format == "srt":
        return to_srt(result["chunks"])
    elif format == "vtt":
        return to_vtt(result["chunks"])

    # Default: return the full text
    return result["text"]
