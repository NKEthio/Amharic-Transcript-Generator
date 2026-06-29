from amharic_asr.transcribe import format_timestamp, to_srt, to_vtt

def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(1.5) == "00:00:01,500"
    assert format_timestamp(61.234) == "00:01:01,234"
    assert format_timestamp(3661.001) == "01:01:01,001"

    assert format_timestamp(0, decimal_marker=".") == "00:00:00.000"
    assert format_timestamp(1.5, decimal_marker=".") == "00:00:01.500"

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

def test_to_vtt_with_none_end():
    chunks = [
        {"timestamp": (10.0, None), "text": "መጨረሻ"},
    ]
    expected = (
        "WEBVTT\n"
        "\n"
        "00:00:10.000 --> 00:00:10.000\n"
        "መጨረሻ\n"
    )
    result = to_vtt(chunks)
    assert result == expected
