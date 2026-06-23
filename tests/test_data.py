import pytest
from amharic_asr.data import normalize_amharic_text

def test_normalize_homophones():
    # Test 'h' sounds
    assert normalize_amharic_text("ሐመልማል") == "ሀመልማል"
    assert normalize_amharic_text("ኀይል") == "ሀይል"
    assert normalize_amharic_text("ኸረ") == "ሀረ"

    # Test 's' sounds
    assert normalize_amharic_text("ሠላም") == "ሰላም"

    # Test 'a' sounds
    assert normalize_amharic_text("ዐለም") == "አለም"

    # Test 'ts' sounds
    assert normalize_amharic_text("ፀሐይ") == "ጸሀይ"

def test_remove_amharic_punctuation():
    text = "ሰላም ነው፨ እንዴት ነህ፧ ደህና ነኝ።"
    normalized = normalize_amharic_text(text)
    assert "፨" not in normalized
    assert "፧" not in normalized
    assert "።" not in normalized
    assert normalized == "ሰላም ነው እንዴት ነህ ደህና ነኝ"

def test_remove_standard_punctuation():
    text = "Hello! ሰላም ነው? (እንዴት ነህ)"
    normalized = normalize_amharic_text(text)
    # "Hello" is removed because it's not in the Ethiopic Unicode range
    # "!" and "?" and "()" are removed
    assert normalized == "ሰላም ነው እንዴት ነህ"

def test_whitespace_cleanup():
    text = "  ሰላም    ነው   "
    assert normalize_amharic_text(text) == "ሰላም ነው"

def test_empty_input():
    assert normalize_amharic_text("") == ""
    assert normalize_amharic_text(None) == ""
