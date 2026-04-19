# Sawtooth-to-Cello: Neural Audio Synthesis with U-Net and HiFi-GAN

A deep learning pipeline for **timbre transfer**: converting synthetic sawtooth waveforms into realistic cello sounds
while preserving the original melody, timing, and pitch.

## Overview

MIDI notes are synthesized as sawtooth waves, then transformed into cello audio through a two-stage neural pipeline:

1. **U-Net** — maps sawtooth mel spectrograms to cello mel spectrograms
2. **HiFi-GAN** — converts the predicted mel spectrogram back to a waveform

Training data comes from the [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) classical piano MIDI
corpus, rendered with FluidSynth using a cello SoundFont as supervision targets.

## Architecture

| Component | Role |
|-----------|------|
| U-Net (encoder-decoder + skip connections) | Mel-to-mel spectrogram translation |
| PatchGAN Discriminator | Adversarial sharpening of spectrogram predictions |
| HiFi-GAN (UNIVERSAL_V1, fine-tuned) | Neural vocoder: mel spectrogram to waveform |

**U-Net details:**
- Encoder: 3 downsampling stages (64 → 128 → 256 → 512 channels) with residual blocks
- Bottleneck: 2 residual blocks with 30% dropout
- Decoder: 3 upsampling stages with skip-connection concatenation
- Output: single-channel mel spectrogram (80 bins × 128 frames)

**Training loss:**

$$\mathcal{L}_G = 100 \cdot \|\hat{M} - M_{\text{target}}\|_1 + \mathcal{L}_{\text{GAN}}(\hat{M})$$

## Key Techniques

- Monophonic melody extraction from polyphonic MIDI for cleaner synthesis targets
- Adversarial training (L1 + GAN loss) for perceptually sharp mel predictions
- Overlap-add inference with Hann windowing for seamless long-form generation
- HiFi-GAN fine-tuned on cello audio using multi-resolution spectral loss (FFT scales: 256, 512, 1024)
- Pre-rendered mel spectrogram pairs saved to Google Drive for efficient training

## Usage

This project runs entirely in Google Colab with no local hardware requirements.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eyalab1/Sawtooth-to-Cello/blob/main/Saw_To_Cello.ipynb)

1. Open the notebook in Colab
2. Mount Google Drive (checkpoints and pre-rendered data are saved to `/content/drive/MyDrive/CelloProject/`)
3. Run cells sequentially — environment setup, data rendering, training, and inference are all self-contained

## Dependencies

Installed automatically by the notebook:

- `fluidsynth` (apt) — MIDI-to-audio rendering
- `pretty_midi`, `librosa`, `soundfile`, `pyfluidsynth`
- `torch`, `torchaudio`
- [HiFi-GAN](https://github.com/jik876/hifi-gan) (cloned and patched for PyTorch 1.8+ / librosa compatibility)

## Related Work

| Work | Method |
|------|--------|
| Jansson et al. (2017) | U-Net frequency masking for vocal separation |
| Mor et al. (2019) | Unsupervised instrument domain translation |
| Engel et al. (2020) — DDSP | Differentiable DSP for violin/flute |
| **This work** | Supervised U-Net + HiFi-GAN + PatchGAN for sawtooth-to-cello |
