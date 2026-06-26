import pytest
from amharic_asr.data import normalize_amharic

def test_normalize_amharic_punctuation():
    text = "ሰላም፧ እንዴት ነህ፥ ደህና ነኝ።"
    # \u1361-1368 covers ፡፣፤፥፦፧።፤
    # We expect punctuation to be replaced by space and then extra spaces collapsed.
    normalized = normalize_amharic(text)
    assert "፧" not in normalized
    assert "፥" not in normalized
    assert "።" not in normalized
    assert normalized == "ሰላም እንዴት ነህ ደህና ነኝ"

def test_normalize_amharic_homophones_ha():
    # ሀ, ሐ, ኀ -> ሀ
    text = "ሐሳብ ኀይል ሀገር"
    normalized = normalize_amharic(text)
    assert normalized == "ሀሳብ ሀይል ሀገር"

def test_normalize_amharic_homophones_se():
    # ሰ, ሠ -> ሰ
    text = "ሠላም ሰላም"
    normalized = normalize_amharic(text)
    assert normalized == "ሰላም ሰላም"

def test_normalize_amharic_homophones_a():
    # አ, ዐ -> አ
    text = "ዐለም አገር"
    normalized = normalize_amharic(text)
    assert normalized == "አለም አገር"

def test_normalize_amharic_homophones_tse():
    # ጸ, ፀ -> ጸ
    text = "ፀሐይ ጸሐይ"
    normalized = normalize_amharic(text)
    assert normalized == "ጸሀይ ጸሀይ"

def test_normalize_amharic_extra_spaces():
    text = "  ሰላም    እንዴት   ነህ  "
    normalized = normalize_amharic(text)
    assert normalized == "ሰላም እንዴት ነህ"

def test_normalize_amharic_empty():
    assert normalize_amharic("") == ""
    assert normalize_amharic(None) == ""
