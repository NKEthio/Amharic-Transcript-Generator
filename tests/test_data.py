from amharic_asr.data import normalize_amharic_text


def test_normalize_amharic_text_homophones():
    # Test cases for each order of homophones
    # ሀ, ሐ, ኀ -> ሀ
    assert normalize_amharic_text("ሐመረ ኀይል") == "ሀመረ ሀይል"
    # ሰ, ሠ -> ሰ
    assert normalize_amharic_text("ሠላም ሰላም") == "ሰላም ሰላም"
    # አ, ዐ -> አ
    assert normalize_amharic_text("አለም ዐለም") == "አለም አለም"
    # ጸ, ፀ -> ጸ
    assert normalize_amharic_text("ፀሀይ ጸሀይ") == "ጸሀይ ጸሀይ"

    # All 7 orders for ሀ homophones
    assert normalize_amharic_text("ሐሑሒሓሔሕሖ") == "ሀሁሂሃሄህሆ"
    assert normalize_amharic_text("ኀኁኂኃኄኅኆ") == "ሀሁሂሃሄህሆ"

    # All 7 orders for ሰ homophones
    assert normalize_amharic_text("ሠሡሢሣሤሥሦ") == "ሰሱሲሳሴስሶ"

    # All 7 orders for አ homophones
    assert normalize_amharic_text("ዐዑዒዓዔዕዖ") == "አኡኢኣኤእኦ"

    # All 7 orders for ጸ homophones
    assert normalize_amharic_text("ፀፁፂፃፄፅፆ") == "ጸጹጺጻጼጽጾ"


def test_normalize_amharic_text_punctuation():
    # Ge'ez punctuation removal
    assert normalize_amharic_text("ሰላም፡ለእናንተ፡ይሁን።") == "ሰላም ለእናንተ ይሁን"
    assert normalize_amharic_text("እንዴት፤ነህ፧") == "እንዴት ነህ"

    # Standard punctuation removal
    assert normalize_amharic_text("ሰላም, እንዴት ነህ?") == "ሰላም እንዴት ነህ"

    # Whitespace normalization
    assert normalize_amharic_text("ሰላም    እንዴት   ነህ ") == "ሰላም እንዴት ነህ"


def test_normalize_amharic_text_mixed():
    input_text = "ሠላም፡ለዓለም፡ይሁን፤ ፀሐይ፡ወጣች።"
    # ዓ (order 4) should normalize to ኣ (order 4)
    expected_text = "ሰላም ለኣለም ይሁን ጸሀይ ወጣች"
    assert normalize_amharic_text(input_text) == expected_text
