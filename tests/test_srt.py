from amharic_asr.transcribe import format_timestamp, to_srt, to_vtt

def test_format_timestamp_srt():
    assert format_timestamp(0.0) == "00:00:00,000"
    assert format_timestamp(3661.123) == "01:01:01,123"
    assert format_timestamp(59.999) == "00:00:59,999"

def test_format_timestamp_vtt():
    assert format_timestamp(0.0, vtt=True) == "00:00:00.000"
    assert format_timestamp(3661.123, vtt=True) == "01:01:01.123"
    assert format_timestamp(59.999, vtt=True) == "00:00:59.999"

def test_to_srt():
    chunks = [
        {"timestamp": (0.0, 2.0), "text": "ሰላም"},
        {"timestamp": (2.0, 4.0), "text": "እንዴት ነህ"},
    ]
    srt = to_srt(chunks)
    expected = (
        "1\n"
        "00:00:00,000 --> 00:00:02,000\n"
        "ሰላም\n"
        "\n"
        "2\n"
        "00:00:02,000 --> 00:00:04,000\n"
        "እንዴት ነህ\n"
    )
    assert srt.strip() == expected.strip()

def test_to_vtt():
    chunks = [
        {"timestamp": (0.0, 2.0), "text": "ሰላም"},
        {"timestamp": (2.0, 4.5), "text": "እንዴት ነህ"},
    ]
    vtt = to_vtt(chunks)
    expected = (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:02.000\n"
        "ሰላም\n"
        "\n"
        "00:00:02.000 --> 00:00:04.500\n"
        "እንዴት ነህ\n"
    )
    assert vtt.strip() == expected.strip()

def test_to_srt_missing_end():
    chunks = [
        {"timestamp": (10.0, None), "text": "መጨረሻ"},
    ]
    srt = to_srt(chunks)
    assert "00:00:10,000 --> 00:00:11,000" in srt
