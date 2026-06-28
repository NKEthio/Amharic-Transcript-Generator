#!/usr/bin/env python3
import os
import sys
import gradio as gr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from amharic_asr.transcribe import transcribe_audio, to_srt, to_vtt

def process_audio(audio_path, model_dir, output_format):
    """
    Processes audio and returns transcript in requested format.
    """
    if not audio_path:
        return "Please upload an audio file."
    if not model_dir:
        return "Please provide a model directory."
    if not os.path.exists(model_dir):
        return f"Model directory not found: {model_dir}"

    try:
        # Determine if we need timestamps
        return_timestamps = output_format in ["srt", "vtt"]

        result = transcribe_audio(
            model_dir,
            audio_path,
            return_timestamps=return_timestamps
        )

        if output_format == "srt":
            return to_srt(result.get("chunks", []))
        elif output_format == "vtt":
            return to_vtt(result.get("chunks", []))
        else:
            return result["text"]

    except Exception as e:
        return f"Error during transcription: {str(e)}"

def main():
    demo = gr.Interface(
        fn=process_audio,
        inputs=[
            gr.Audio(type="filepath", label="Upload Amharic Audio"),
            gr.Textbox(
                label="Model Directory",
                value="outputs/amharic-whisper-small-ft",
                placeholder="Path to fine-tuned model"
            ),
            gr.Radio(
                choices=["txt", "srt", "vtt"],
                value="txt",
                label="Output Format"
            )
        ],
        outputs=gr.Textbox(label="Transcript"),
        title="Amharic ASR Web Interface",
        description="Upload an Amharic audio file to generate its transcript in various formats using a fine-tuned Whisper model."
    )

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
