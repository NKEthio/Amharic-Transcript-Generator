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

def process_audio(audio_path, model_dir, output_format, chunk_length_s, task):
    """
    Handles audio transcription and prepares the result for display and download.
    """
    if not audio_path:
        return "Please upload an audio file.", None
    if not model_dir:
        return "Please provide a model directory.", None
    if not os.path.exists(model_dir):
        return f"Model directory not found: {model_dir}", None

    try:
        # Generate transcript using the core transcription logic
        transcript = transcribe_audio(
            model_dir,
            audio_path,
            format=output_format,
            chunk_length_s=chunk_length_s,
            task=task
        )

        # Save to a temporary file for download
        suffix = f".{output_format}"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='w', encoding='utf-8')
        temp_file.write(transcript)
        temp_file.close()

        return transcript, temp_file.name
    except Exception as e:
        return f"Error during transcription: {str(e)}", None

def main():
    # Using gr.Blocks for a more customized and flexible interface
    with gr.Blocks(title="Amharic ASR Web Interface") as demo:
        gr.Markdown("# Amharic ASR Web Interface")
        gr.Markdown("Upload an Amharic audio file to generate its transcript using a fine-tuned Whisper model.")

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Upload Amharic Audio")
                model_dir_input = gr.Textbox(
                    label="Model Directory",
                    value="outputs/amharic-whisper-small-ft",
                    placeholder="Path to fine-tuned model"
                )
                with gr.Row():
                    format_input = gr.Radio(
                        choices=["txt", "srt", "vtt"],
                        value="txt",
                        label="Output Format"
                    )
                    task_input = gr.Radio(
                        choices=["transcribe", "translate"],
                        value="transcribe",
                        label="Task"
                    )
                    chunk_input = gr.Slider(
                        minimum=5,
                        maximum=60,
                        value=30,
                        step=5,
                        label="Chunk Length (seconds)"
                    )

                submit_btn = gr.Button("Generate Transcript", variant="primary")

            with gr.Column():
                transcript_output = gr.Textbox(label="Transcript", lines=15)
                file_output = gr.File(label="Download Transcript")

        # Link the button to the processing function
        submit_btn.click(
            fn=process_audio,
            inputs=[audio_input, model_dir_input, format_input, chunk_input, task_input],
            outputs=[transcript_output, file_output]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
