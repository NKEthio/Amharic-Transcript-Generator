from unittest.mock import MagicMock, patch
from amharic_asr.transcribe import format_timestamp, to_srt, to_vtt, transcribe_audio

def test_format_timestamp():
    # Test SRT format (default)
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(1.5) == "00:00:01,500"
    assert format_timestamp(61.234) == "00:01:01,234"
    assert format_timestamp(3661.001) == "01:01:01,001"

    # Test VTT format
    assert format_timestamp(0, format="vtt") == "00:00:00.000"
    assert format_timestamp(1.5, format="vtt") == "00:00:01.500"

def test_to_srt():
    chunks = [
        {"timestamp": (0.0, 2.5), "text": "ሰላም"},
        {"timestamp": (2.5, 5.0), "text": "እንዴት ነህ"},
    ]
    expected = (
        "1\n"
        "00:00:00,000 --> 00:00:02,500\n"
        "ሰላም\n"
        "\n"
        "2\n"
        "00:00:02,500 --> 00:00:05,000\n"
        "እንዴት ነህ\n"
    )
    result = to_srt(chunks)
    assert result == expected

def test_to_vtt():
    chunks = [
        {"timestamp": (0.0, 2.5), "text": "ሰላም"},
        {"timestamp": (2.5, 5.0), "text": "እንዴት ነህ"},
    ]
    expected = (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:02.500\n"
        "ሰላም\n"
        "\n"
        "00:00:02.500 --> 00:00:05.000\n"
        "እንዴት ነህ\n"
    )
    result = to_vtt(chunks)
    assert result == expected

def test_to_srt_with_none_end():
    chunks = [
        {"timestamp": (10.0, None), "text": "መጨረሻ"},
    ]
    expected = (
        "1\n"
        "00:00:10,000 --> 00:00:10,000\n"
        "መጨረሻ\n"
    )
    result = to_srt(chunks)
    assert result == expected

@patch("amharic_asr.transcribe.librosa.load")
@patch("amharic_asr.transcribe.pipeline")
@patch("amharic_asr.transcribe.torch.cuda.is_available")
def test_transcribe_audio_interface(mock_cuda, mock_pipeline, mock_load):
    mock_cuda.return_value = False
    mock_asr = MagicMock()
    mock_pipeline.return_value = mock_asr
    mock_load.return_value = (MagicMock(), 16000)

    # Mocking asr returns
    mock_asr.return_value = {"text": "ሰላም", "chunks": [{"timestamp": (0.0, 1.0), "text": "ሰላም"}]}

    # Test TXT format
    res = transcribe_audio("model", "audio.wav", format="txt")
    assert res == "ሰላም"
    mock_pipeline.assert_called_with(
        "automatic-speech-recognition",
        model="model",
        device=-1,
        chunk_length_s=30
    )
    mock_asr.assert_called()
    call_args = mock_asr.call_args
    assert call_args[1]["return_timestamps"] is False
    assert call_args[1]["generate_kwargs"] == {"language": "amharic", "task": "transcribe"}

    # Test SRT format
    res = transcribe_audio("model", "audio.wav", format="srt")
    assert "00:00:00,000 --> 00:00:01,000" in res
    mock_asr.assert_called()
    call_args = mock_asr.call_args
    assert call_args[1]["return_timestamps"] is True
    assert call_args[1]["generate_kwargs"] == {"language": "amharic", "task": "transcribe"}
