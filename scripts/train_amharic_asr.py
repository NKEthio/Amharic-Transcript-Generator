#!/usr/bin/env python3
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from amharic_asr.config import load_training_config
from amharic_asr.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a foundation model for Amharic ASR.")
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    args = parser.parse_args()

    config = load_training_config(args.config)
    train_model(config)


if __name__ == "__main__":
    main()
