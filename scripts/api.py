import os
import sys
import tempfile
import torch
from typing import Optional, Any
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse

# Add src to sys.path for local imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from amharic_asr.transcribe import transcribe_audio, load_transcription_pipeline

app = FastAPI(
    title="Amharic ASR API",
    description="A REST API for Amharic Speech-to-Text using Whisper",
    version="1.0.0"
)

# Pipeline cache to avoid reloading the model on every request
class PipelineCache:
    def __init__(self):
        self.model_dir: Optional[str] = None
        self.chunk_length_s: Optional[int] = None
        self.pipeline: Any = None

    def get_pipeline(self, model_dir: str, chunk_length_s: int):
        if self.pipeline is None or self.model_dir != model_dir or self.chunk_length_s != chunk_length_s:
            print(f"Loading pipeline for {model_dir} with chunk_length_s={chunk_length_s}...")
            self.pipeline = load_transcription_pipeline(
                model_dir=model_dir,
                chunk_length_s=chunk_length_s
            )
            self.model_dir = model_dir
            self.chunk_length_s = chunk_length_s
        return self.pipeline

cache = PipelineCache()

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model_dir: str = Query("outputs/amharic-whisper-small-ft", description="Path to the fine-tuned model directory or HF model ID"),
    format: str = Query("txt", enum=["txt", "srt", "vtt"], description="Output transcript format"),
    task: str = Query("transcribe", enum=["transcribe", "translate"], description="Transcription or translation task"),
    chunk_length_s: int = Query(30, ge=5, le=60, description="Audio chunk length in seconds")
):
    """
    Upload an audio file and get its Amharic transcription.
    """
    # Create a temporary file to store the uploaded audio
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_audio:
        content = await file.read()
        tmp_audio.write(content)
        tmp_audio_path = tmp_audio.name

    try:
        # Get cached pipeline or load it
        asr_pipeline = cache.get_pipeline(model_dir, chunk_length_s)

        transcript = transcribe_audio(
            model_dir=model_dir,
            audio_path=tmp_audio_path,
            format=format,
            task=task,
            chunk_length_s=chunk_length_s,
            asr_pipeline=asr_pipeline
        )

        return {
            "filename": file.filename,
            "format": format,
            "task": task,
            "transcript": transcript
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary audio file
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

@app.get("/health")
def health_check():
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
