from unittest.mock import MagicMock, patch
from amharic_asr.transcribe import load_transcription_pipeline, transcribe_audio

@patch("amharic_asr.transcribe.pipeline")
@patch("amharic_asr.transcribe.torch.cuda.is_available")
def test_load_transcription_pipeline(mock_cuda, mock_pipeline):
    mock_cuda.return_value = False

    load_transcription_pipeline("model_dir", device=-1, chunk_length_s=30)

    mock_pipeline.assert_called_once_with(
        "automatic-speech-recognition",
        model="model_dir",
        device=-1,
        chunk_length_s=30
    )

@patch("amharic_asr.transcribe.librosa.load")
@patch("amharic_asr.transcribe.pipeline")
def test_transcribe_audio_with_reuse(mock_pipeline, mock_load):
    mock_load.return_value = (MagicMock(), 16000)
    mock_asr = MagicMock()
    mock_asr.return_value = {"text": "result"}

    # Use pre-loaded pipeline
    res = transcribe_audio("model", "audio.wav", asr_pipeline=mock_asr)

    assert res == "result"
    # Should NOT call pipeline() again
    mock_pipeline.assert_not_called()
    mock_asr.assert_called_once()
