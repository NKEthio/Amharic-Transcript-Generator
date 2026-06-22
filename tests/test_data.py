import pytest
from amharic_asr.data import normalize_amharic


def test_normalize_ha_family():
    # Test mapping of ሐ and ኀ families to ሀ
    assert normalize_amharic("ሐሑሒሓሔሕሖ") == "ሀሁሂሃሄህሆ"
    assert normalize_amharic("ኀኁኂኃኄኅኆ") == "ሀሁሂሃሄህሆ"


def test_normalize_se_family():
    # Test mapping of ሠ family to ሰ
    assert normalize_amharic("ሠሡሢሣሤሥሦ") == "ሰሱሲሳሴስሶ"


def test_normalize_a_family():
    # Test mapping of ዐ family to አ
    assert normalize_amharic("ዐዑዒዓዔዕዖ") == "አኡኢኣኤእኦ"


def test_normalize_tse_family():
    # Test mapping of ፀ family to ጸ
    assert normalize_amharic("ፀፁፂፃፄፅፆ") == "ጸጹጺጻጼጽጾ"


def test_remove_punctuation():
    # Test removal of Ge'ez and standard punctuation
    text = "ሰላም፡ አለህ? አዎ፣ አለሁ።"
    expected = "ሰላም አለህ አዎ አለሁ"
    assert normalize_amharic(text) == expected


def test_whitespace_cleanup():
    # Test cleanup of extra spaces and trimming
    text = "  ሰላም    አለህ   "
    expected = "ሰላም አለህ"
    assert normalize_amharic(text) == expected


def test_mixed_normalization():
    # Test a full sentence with mixed characters and punctuation
    text = "ሐምሌ ፲፱፻፹፰ ዓ.ም. ፀሐይ በሥራ ላይ ነበረች።"
    # ሐ -> ሀ, ፹ -> no change (not in mapping yet, only Ge'ez punct), ዓ -> ኣ, . -> removed, ፀ -> ጸ, ሐ -> ሀ, ሥ -> ስ, ። -> removed
    # Note: ፹ is a number, not Ge'ez punctuation mark in the regex.
    # . is Western punct.
    expected = "ሀምሌ ፲፱፻፹፰ ኣም ጸሀይ በስራ ላይ ነበረች"
    assert normalize_amharic(text) == expected
