import torch
from transformers import pipeline


def format_timestamp(seconds: float, srt: bool = False) -> str:
    """
    Converts seconds into a subtitle timestamp format (HH:MM:SS,mmm or HH:MM:SS.mmm).
    """
    td_hours = int(seconds // 3600)
    td_mins = int((seconds % 3600) // 60)
    td_secs = int(seconds % 60)
    td_millis = int(round((seconds % 1) * 1000))

    separator = "," if srt else "."
    return f"{td_hours:02d}:{td_mins:02d}:{td_secs:02d}{separator}{td_millis:03d}"


def to_srt(chunks: list[dict]) -> str:
    """
    Converts ASR chunks with timestamps into SRT format.
    """
    srt_output = []
    for i, chunk in enumerate(chunks, 1):
        start, end = chunk["timestamp"]
        # Handle cases where end might be None (last chunk)
        if end is None:
            # Estimate end as start + 5 seconds or similar, or just leave it
            end = start + 5.0  # Simple fallback

        srt_output.append(str(i))
        srt_output.append(f"{format_timestamp(start, srt=True)} --> {format_timestamp(end, srt=True)}")
        srt_output.append(chunk["text"].strip())
        srt_output.append("")
    return "\n".join(srt_output)


def to_vtt(chunks: list[dict]) -> str:
    """
    Converts ASR chunks with timestamps into WebVTT format.
    """
    vtt_output = ["WEBVTT", ""]
    for chunk in chunks:
        start, end = chunk["timestamp"]
        if end is None:
            end = start + 5.0

        vtt_output.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        vtt_output.append(chunk["text"].strip())
        vtt_output.append("")
    return "\n".join(vtt_output)


def transcribe_audio(
    model_dir: str,
    audio_path: str,
    device: int | None = None,
    chunk_length_s: int = 30,
    return_timestamps: bool = False,
) -> dict:
    """
    Transcribes Amharic audio using a fine-tuned Whisper model.
    Returns a dictionary containing 'text' and optionally 'chunks'.
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        device=device,
        chunk_length_s=chunk_length_s,
    )

    # generate_kwargs for Amharic
    generate_kwargs = {"language": "am", "task": "transcribe"}

    result = asr(
        audio_path,
        return_timestamps=return_timestamps,
        generate_kwargs=generate_kwargs
    )
    return result
