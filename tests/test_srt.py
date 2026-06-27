import pytest
from amharic_asr.transcribe import format_timestamp, to_srt

def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(1.5) == "00:00:01,500"
    assert format_timestamp(61.234) == "00:01:01,234"
    assert format_timestamp(3661.005) == "01:01:01,005"

def test_to_srt():
    chunks = [
        {"timestamp": (0.0, 2.0), "text": "ሰላም"},
        {"timestamp": (2.0, 4.5), "text": "እንዴት ነህ?"},
        {"timestamp": (4.5, None), "text": "ደህና ነኝ።"}
    ]
    srt = to_srt(chunks)

    expected_lines = [
        "1",
        "00:00:00,000 --> 00:00:02,000",
        "ሰላም",
        "",
        "2",
        "00:00:02,000 --> 00:00:04,500",
        "እንዴት ነህ?",
        "",
        "3",
        "00:00:04,500 --> 00:00:05,500",
        "ደህና ነኝ።",
        ""
    ]
    expected_srt = "\n".join(expected_lines)
    assert srt == expected_srt
