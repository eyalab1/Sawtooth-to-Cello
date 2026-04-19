# Sawtooth-to-Cello

Sawtooth-to-Cello is a deep-learning timbre-transfer project that converts simple sawtooth audio into cello-like audio while preserving melody and timing.

The full implementation is in the notebook:

- `Saw_To_Cello.ipynb`

## What the notebook builds

The notebook implements an end-to-end pipeline:

1. Generate/parse MIDI musical content
2. Synthesize sawtooth input audio
3. Render cello target audio with FluidSynth + SoundFont
4. Convert both signals to mel spectrograms
5. Train a U-Net to map sawtooth mel → cello mel
6. Improve realism with adversarial training (PatchGAN)
7. Convert predicted mel spectrograms back to waveform with HiFi-GAN

## Model components

- **AudioUNet (Generator):** Mel-to-mel translation from synthetic timbre to cello timbre
- **Patch Discriminator:** Adversarial supervision for sharper, more realistic mel details
- **HiFi-GAN Vocoder:** Neural vocoder used to synthesize final waveform audio from mel spectrograms (fine-tuned for cello)

## Data pipeline

- MIDI source: **MAESTRO v3.0.0 (MIDI)**
- Targets: Cello renderings generated via **FluidSynth** with **FluidR3_GM SoundFont** (program 42)
- Training format: paired mel tensors (sawtooth input, cello target), typically processed in fixed-size windows

## Notebook workflow (high level)

The notebook covers:

- Environment and dependency setup (Colab-style workflow)
- HiFi-GAN clone, patching, and checkpoint loading
- Sanity reconstruction tests
- U-Net definition and quick training smoke test
- MAESTRO download and paired-data pre-rendering
- HiFi-GAN cello fine-tuning
- Full adversarial U-Net training
- Inference on validation/custom MIDI
- Spectrogram comparison and qualitative evaluation

## Requirements (as used in notebook)

- Python 3
- PyTorch
- FluidSynth
- `pyfluidsynth`
- `pretty_midi`
- `librosa`
- `soundfile`
- `matplotlib`

## Running

This repository is currently notebook-first.

1. Open `Saw_To_Cello.ipynb` in Google Colab or Jupyter
2. Run cells in order
3. Provide Google Drive access if using the notebook’s default storage paths

## Repository contents

- `Saw_To_Cello.ipynb` — complete project pipeline, training, and inference
- `README.md` — project overview

## Notes

- The target cello audio is SoundFont-based synthesis, not live recorded cello.
- Training and inference are designed around mel spectrogram consistency with HiFi-GAN settings.
