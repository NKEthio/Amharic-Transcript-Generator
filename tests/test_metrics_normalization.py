import pytest
import jiwer
from amharic_asr.data import normalize_amharic

def test_normalization_impact_on_wer():
    # Homophones should match after normalization
    ref = "ሐሳብ"
    pred = "ሀሳብ"

    # Before normalization, WER should be 1.0 (they are different)
    assert jiwer.wer(ref, pred) == 1.0

    # After normalization, they should be the same
    norm_ref = normalize_amharic(ref)
    norm_pred = normalize_amharic(pred)
    assert norm_ref == norm_pred
    assert jiwer.wer(norm_ref, norm_pred) == 0.0

def test_normalization_removes_punctuation():
    ref = "ሰላም ነህ?"
    pred = "ሰላም ነህ"

    # After normalization, both should be "ሰላም ነህ"
    assert normalize_amharic(ref) == normalize_amharic(pred)
    assert jiwer.wer(normalize_amharic(ref), normalize_amharic(pred)) == 0.0

def test_cer_normalization():
    ref = "ዐለም"
    pred = "አለም"

    norm_ref = normalize_amharic(ref)
    norm_pred = normalize_amharic(pred)

    assert jiwer.cer(norm_ref, norm_pred) == 0.0
