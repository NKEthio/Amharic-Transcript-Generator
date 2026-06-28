from amharic_asr.transcribe import format_timestamp, to_srt

def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(1.5) == "00:00:01,500"
    assert format_timestamp(61.234) == "00:01:01,234"
    assert format_timestamp(3661.001) == "01:01:01,001"

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
    # The to_srt function joins with \n, and our expected string uses \n too.
    # Note: to_srt uses "\n".join(srt_lines) and each text block has \n appended.
    # Let's check the implementation again.

    # srt_lines.append(f"{i}")
    # srt_lines.append(f"{start_str} --> {end_str}")
    # srt_lines.append(f"{text}\n")
    # return "\n".join(srt_lines)

    # For chunk 1:
    # lines = ["1", "00:00:00,000 --> 00:00:02,500", "ሰላም\n"]
    # For chunk 2:
    # lines = ["1", "00:00:00,000 --> 00:00:02,500", "ሰላም\n", "2", "00:00:02,500 --> 00:00:05,000", "እንዴት ነህ\n"]
    # Joined: "1\n00:00:00,000 --> 00:00:02,500\nሰላም\n\n2\n00:00:02,500 --> 00:00:05,000\nእንዴት ነህ\n"

    result = to_srt(chunks)
    assert result == expected

def test_to_srt_with_none_end():
    chunks = [
        {"timestamp": (10.0, None), "text": "መጨረሻ"},
    ]
    # In our implementation, None end is replaced with start
    expected = (
        "1\n"
        "00:00:10,000 --> 00:00:10,000\n"
        "መጨረሻ\n"
    )
    result = to_srt(chunks)
    assert result == expected
