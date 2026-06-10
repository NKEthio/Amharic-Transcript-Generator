# Amharic Transcript Generator

End-to-end Amharic speech-to-text pipeline for fine-tuning foundation models (FMs) on Amharic voice data and generating transcripts.

## What this project includes

- Data loading from CSV manifests (`audio_path`, `transcript`)
- Fine-tuning workflow for `openai/whisper-small`
- Word error rate (WER) evaluation during training
- Inference script for generating Amharic transcripts from audio
- Config-driven training using YAML

## Project structure

```text
configs/
  amharic_whisper_ft.yaml
scripts/
  train_amharic_asr.py
  transcribe_audio.py
src/
  amharic_asr/
    config.py
    data.py
    train.py
    transcribe.py
tests/
  test_config.py
```

## Installation

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset format

Prepare two CSV files:

- `train.csv`
- `validation.csv`

Required columns:

- `audio_path`: absolute or relative path to `.wav/.mp3/...` audio file
- `transcript`: Amharic reference text

Example:

```csv
audio_path,transcript
data/audio/sample1.wav,ሰላም እንዴት ነህ
data/audio/sample2.wav,ይህ የአማርኛ ድምጽ ነው
```

## Fine-tuning an FM for Amharic ASR

1. Edit `configs/amharic_whisper_ft.yaml` with your dataset paths and training settings.
2. Run:

```bash
python scripts/train_amharic_asr.py --config configs/amharic_whisper_ft.yaml
```

Model checkpoints and final artifacts are saved to `outputs/amharic-whisper-small-ft` by default.
Set `preprocessing_num_proc` in config to use more CPU cores during feature preparation.

## Generate transcript from audio

```bash
python scripts/transcribe_audio.py \
  --model-dir outputs/amharic-whisper-small-ft \
  --audio-path data/audio/sample1.wav
```

## Notes

- This repository provides the full mechanism (data -> fine-tune -> evaluate -> inference) for Amharic transcript generation.
- You can switch to another FM by changing `base_model` in the config.