import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from amharic_asr.config import load_training_config


class TestTrainingConfig(unittest.TestCase):
    def test_load_training_config(self):
        content = """
base_model: openai/whisper-small
train_csv: data/train.csv
validation_csv: data/validation.csv
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = f.name

        try:
            cfg = load_training_config(path)
            self.assertEqual(cfg.base_model, "openai/whisper-small")
            self.assertEqual(cfg.train_csv, "data/train.csv")
            self.assertEqual(cfg.validation_csv, "data/validation.csv")
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
