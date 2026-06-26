import pytest
from amharic_asr.data import normalize_amharic_text

def test_normalize_amharic_text_homophones():
    # Test Group 1: ሀ, ሐ, ኀ -> ሀ
    assert normalize_amharic_text("ሐሑሒሓሔሕሖ") == "ሀሁሂሃሄህሆ"
    assert normalize_amharic_text("ኀኁኂኃኄኅኆ") == "ሀሁሂሃሄህሆ"

    # Test Group 2: ሰ, ሠ -> ሰ
    assert normalize_amharic_text("ሠሡሢሣሤሥሦ") == "ሰሱሲሳሴስሶ"

    # Test Group 3: አ, ዐ -> አ
    assert normalize_amharic_text("ዐዑዒዓዔዕዖ") == "አኡኢኣኤእኦ"

    # Test Group 4: ጸ, ፀ -> ጸ
    assert normalize_amharic_text("ፀፁፂፃፄፅፆ") == "ጸጹጺጻጼጽጾ"

def test_normalize_amharic_text_punctuation():
    # Test Amharic and standard punctuation removal
    text = "ሰላም፣ እንዴት ነህ፧"
    assert normalize_amharic_text(text) == "ሰላም እንዴት ነህ"

    text = "ሀሁ (ሂሃ) [ሄህ] {ሆ}"
    assert normalize_amharic_text(text) == "ሀሁ ሂሃ ሄህ ሆ"

    text = "ጥያቄ? መልስ!"
    assert normalize_amharic_text(text) == "ጥያቄ መልስ"

def test_normalize_amharic_text_whitespace():
    # Test whitespace normalization
    assert normalize_amharic_text("  ሰላም   እንዴት  ነህ  ") == "ሰላም እንዴት ነህ"
    assert normalize_amharic_text("ሰላም\tእንዴት\nነህ") == "ሰላም እንዴት ነህ"

def test_normalize_amharic_text_mixed():
    # Test mixed cases
    text = "ሐምሌ ፭ ቀን ፳፻፲፫ ዓ.ም."
    # ሐ -> ሀ, ዓ -> ኣ, . -> space, whitespace normalization
    # Note: ፭ ፳፻፲፫ are Ge'ez numbers, we are not normalizing them currently, just the homophones
    normalized = normalize_amharic_text(text)
    assert "ሀምሌ" in normalized
    assert "ኣ ም" in normalized
