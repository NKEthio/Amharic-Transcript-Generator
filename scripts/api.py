#!/usr/bin/env python3
import os
import sys
import tempfile
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from amharic_asr.transcribe import transcribe_audio, load_transcription_pipeline

app = FastAPI(
    title="Amharic ASR API",
    description="REST API for Amharic speech-to-text transcription using fine-tuned Whisper models.",
    version="1.0.0"
)

class PipelineCache:
    """
    Manages ASR pipeline instances to avoid redundant model loading.
    """
    def __init__(self):
        self.cache = {}

    def get_pipeline(self, model_dir: str, device: Optional[int], chunk_length_s: int) -> Any:
        # Resolve device index to ensure consistent caching
        if device is None:
            resolved_device = 0 if torch.cuda.is_available() else -1
        else:
            resolved_device = device

        key = (model_dir, resolved_device, chunk_length_s)
        if key not in self.cache:
            print(f"Loading model into cache: {model_dir}")
            self.cache[key] = load_transcription_pipeline(
                model_dir=model_dir,
                device=device,
                chunk_length_s=chunk_length_s
            )
        return self.cache[key]

# Initialize global cache
pipeline_cache = PipelineCache()

@app.get("/health")
async def health_check():
    """Returns the health status of the API."""
    return {"status": "healthy", "service": "amharic-asr"}

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    model_dir: str = Form("outputs/amharic-whisper-small-ft"),
    format: str = Form("txt"),
    task: str = Form("transcribe"),
    chunk_length_s: int = Form(30),
    device: Optional[int] = Form(None)
):
    """
    Transcribes the uploaded audio file.
    """
    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail=f"Model directory not found: {model_dir}")

    # Create a temporary file to store the uploaded audio
    tmp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1]) as tmp_audio:
            content = await audio.read()
            tmp_audio.write(content)
            tmp_audio_path = tmp_audio.name

        # Get pipeline from cache
        asr_pipeline = pipeline_cache.get_pipeline(
            model_dir=model_dir,
            device=device,
            chunk_length_s=chunk_length_s
        )

        # Perform transcription
        transcript = transcribe_audio(
            model_dir=model_dir,
            audio_path=tmp_audio_path,
            device=device,
            chunk_length_s=chunk_length_s,
            format=format,
            task=task,
            asr_pipeline=asr_pipeline
        )

        return {
            "filename": audio.filename,
            "transcript": transcript,
            "format": format,
            "task": task
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary audio file
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
