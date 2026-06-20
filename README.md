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
  --audio-path data/audio/sample1.wav \
  --chunk-length-s 30
```

## Notes

- This repository provides the full mechanism (data -> fine-tune -> evaluate -> inference) for Amharic transcript generation.
- You can switch to another FM by changing `base_model` in the config.

## Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-colab/colab-tools/blob/master/samples/colab-badge.ipynb)

You can run this project easily on Google Colab using the provided notebook:

1. Open `amharic_asr_colab.ipynb` in this repo.
2. Click on the "Open in Colab" button (if viewing on GitHub) or upload the `.ipynb` file to [Google Colab](https://colab.research.google.com/).
3. Follow the instructions in the notebook to mount Google Drive, install dependencies, and start training or transcription.

Alternatively, you can use the standalone script:

```bash
# Training
python amharic_asr_standalone.py train --config configs/amharic_whisper_ft.yaml

# Transcription
python amharic_asr_standalone.py transcribe --model-dir outputs/amharic-whisper-small-ft --audio-path data/audio/sample1.wav
```