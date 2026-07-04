from unittest.mock import MagicMock, patch
import os
import sys

# Ensure src is in sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from amharic_asr.transcribe import transcribe_audio, load_transcription_pipeline

@patch("amharic_asr.transcribe.librosa.load")
@patch("amharic_asr.transcribe.pipeline")
@patch("amharic_asr.transcribe.torch.cuda.is_available")
def test_pipeline_reuse_logic(mock_cuda, mock_pipeline, mock_load):
    mock_cuda.return_value = False
    mock_asr = MagicMock()
    mock_pipeline.return_value = mock_asr
    mock_load.return_value = (MagicMock(), 16000)
    mock_asr.return_value = {"text": "test"}

    model_dir = "mock_model"
    audio_path = "mock_audio.wav"

    # 1. Load pipeline manually
    asr_pipeline = load_transcription_pipeline(model_dir)
    assert mock_pipeline.call_count == 1

    # 2. Call transcribe_audio with the pipeline
    res1 = transcribe_audio(model_dir, audio_path, asr_pipeline=asr_pipeline)
    assert res1 == "test"

    # 3. Call transcribe_audio again with the same pipeline
    res2 = transcribe_audio(model_dir, audio_path, asr_pipeline=asr_pipeline)
    assert res2 == "test"

    # Verify that pipeline was ONLY called once (during load_transcription_pipeline)
    # and NOT again inside transcribe_audio
    assert mock_pipeline.call_count == 1
    assert mock_asr.call_count == 2

@patch("amharic_asr.transcribe.librosa.load")
@patch("amharic_asr.transcribe.pipeline")
@patch("amharic_asr.transcribe.torch.cuda.is_available")
def test_transcribe_audio_loads_if_missing(mock_cuda, mock_pipeline, mock_load):
    mock_cuda.return_value = False
    mock_asr = MagicMock()
    mock_pipeline.return_value = mock_asr
    mock_load.return_value = (MagicMock(), 16000)
    mock_asr.return_value = {"text": "test"}

    model_dir = "mock_model"
    audio_path = "mock_audio.wav"

    # Call transcribe_audio WITHOUT providing a pipeline
    res = transcribe_audio(model_dir, audio_path)

    assert res == "test"
    # Should have loaded the pipeline automatically
    assert mock_pipeline.call_count == 1
    assert mock_asr.call_count == 1
