# Conditional GAN for Audio Synthesis 🎶

This repository contains the code for **Task 2: A deep learning-based generative model for audio synthesis**. The project implements a Conditional Generative Adversarial Network (CGAN) using PyTorch and `torchaudio` to generate realistic, category-specific audio clips. The model is trained on log-mel spectrograms and can produce novel audio samples for any of the predefined categories it was trained on.

This implementation is designed to be run in a Google Colab environment, leveraging its free GPU resources for efficient training.

## 🚀 Features

* **Conditional Generation**: The GAN generates audio conditioned on a specific class label, allowing for targeted audio synthesis.
* **Spectrogram-Based**: The model operates in the frequency domain by generating log-mel spectrograms, a robust representation for audio data.
* **End-to-End Training**: The provided script (`gan_audio.py`) handles everything from data loading and preprocessing to model training and sample generation.
* **Sample Generation**: At the end of each epoch, the model saves generated audio samples (`.wav`) and plots of their corresponding spectrograms.
* **Google Colab Ready**: Includes necessary setup for mounting Google Drive to access datasets.

---

## 🏗️ Model Architecture

The project consists of two main neural network components, following the standard GAN architecture.

### 1. Generator (The Forger 🎨)
The `CGAN_Generator` takes a random noise vector (from the latent space) and a one-hot encoded class label as input. It then uses a series of `ConvTranspose2d` layers to upsample this input into a full-sized log-mel spectrogram ($128 \times 512$ dimensions) that mimics a real audio sample of the specified category.

### 2. Discriminator (The Detective 🕵️)
The `CGAN_Discriminator` takes a log-mel spectrogram and a one-hot encoded class label as input. It concatenates the spectrogram with a learned embedding of the label. Then, it uses a series of `Conv2d` layers to downsample the input and outputs a single value (logit) indicating whether the spectrogram is real or fake *for that particular class*.

---

## 🔧 Setup and Installation

### Prerequisites
* A Google Account (for Google Colab and Google Drive).
* Your audio dataset organized by category.

### 1. Dataset Structure
Before running the script, you must organize your audio files (`.wav`) into a specific directory structure on your Google Drive. The script expects a root folder containing subfolders for each audio category.

For example, if your base path is `drive/MyDrive/organized_dataset/`, your training data should be structured as follows:

```
drive/MyDrive/organized_dataset/
└── train/
    ├── category_1/
    │   ├── audio_001.wav
    │   ├── audio_002.wav
    │   └── ...
    ├── category_2/
    │   ├── sound_A.wav
    │   ├── sound_B.wav
    │   └── ...
    └── category_3/
        ├── sample_x.wav
        ├── sample_y.wav
        └── ...
```


## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
