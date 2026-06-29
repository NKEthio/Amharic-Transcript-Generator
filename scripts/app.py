#!/usr/bin/env python3
import os
import sys
import gradio as gr
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from amharic_asr.transcribe import transcribe_audio

def process_audio(audio_path, model_dir, chunk_length_s, output_format):
    if not audio_path:
        return "Please upload an audio file.", None
    if not model_dir:
        return "Please provide a model directory.", None
    if not os.path.exists(model_dir):
        return f"Model directory not found: {model_dir}", None

    try:
        transcript = transcribe_audio(
            model_dir,
            audio_path,
            chunk_length_s=chunk_length_s,
            format=output_format
        )

        # Create a temporary file for download
        suffix = f".{output_format}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='w', encoding='utf-8') as tmp:
            tmp.write(transcript)
            tmp_path = tmp.name

        return transcript, tmp_path
    except Exception as e:
        return f"Error during transcription: {str(e)}", None

def main():
    with gr.Blocks(title="Amharic ASR Professional") as demo:
        gr.Markdown("# 🇪🇹 Amharic ASR Professional")
        gr.Markdown("Upload an Amharic audio file to generate its transcript using a fine-tuned Whisper model.")

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Upload Amharic Audio")
                model_dir = gr.Textbox(
                    label="Model Directory",
                    value="outputs/amharic-whisper-small-ft",
                    placeholder="Path to fine-tuned model"
                )
                with gr.Row():
                    chunk_length = gr.Slider(
                        minimum=5,
                        maximum=60,
                        value=30,
                        step=5,
                        label="Chunk Length (seconds)"
                    )
                    format_dropdown = gr.Dropdown(
                        choices=["txt", "srt", "vtt"],
                        value="txt",
                        label="Output Format"
                    )
                transcribe_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column():
                transcript_output = gr.Textbox(label="Transcript Preview", lines=15)
                file_output = gr.File(label="Download Transcript File")

        transcribe_btn.click(
            fn=process_audio,
            inputs=[audio_input, model_dir, chunk_length, format_dropdown],
            outputs=[transcript_output, file_output]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
