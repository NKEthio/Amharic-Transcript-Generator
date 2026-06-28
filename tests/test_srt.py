from amharic_asr.transcribe import format_timestamp, to_srt, to_vtt

def test_format_timestamp():
    # Test VTT format (default)
    assert format_timestamp(0) == "00:00:00.000"
    assert format_timestamp(3661.123) == "01:01:01.123"

    # Test SRT format
    assert format_timestamp(0, srt=True) == "00:00:00,000"
    assert format_timestamp(3661.123, srt=True) == "01:01:01,123"

def test_to_srt():
    chunks = [
        {"timestamp": (0.0, 2.5), "text": "ሰላም"},
        {"timestamp": (2.5, 5.0), "text": "እንዴት ነህ"}
    ]
    srt = to_srt(chunks)
    assert "1" in srt
    assert "00:00:00,000 --> 00:00:02,500" in srt
    assert "ሰላም" in srt
    assert "2" in srt
    assert "00:00:02,500 --> 00:00:05,000" in srt
    assert "እንዴት ነህ" in srt

def test_to_srt_none_end():
    chunks = [{"timestamp": (10.0, None), "text": "መጨረሻ"}]
    srt = to_srt(chunks)
    assert "00:00:10,000 --> 00:00:15,000" in srt
    assert "መጨረሻ" in srt

def test_to_vtt():
    chunks = [
        {"timestamp": (0.0, 2.5), "text": "ሰላም"},
        {"timestamp": (2.5, 5.0), "text": "እንዴት ነህ"}
    ]
    vtt = to_vtt(chunks)
    assert "WEBVTT" in vtt
    assert "00:00:00.000 --> 00:00:02.500" in vtt
    assert "ሰላም" in vtt
    assert "00:00:02.500 --> 00:00:05.000" in vtt
    assert "እንዴት ነህ" in vtt
