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

### 2. Installation
The required Python packages can be installed using `pip`. The first cell in the provided Colab script handles this automatically:

```bash
!pip install torch torchaudio torchvision transformers
```

---

## ▶️ How to Run

1.  **Upload Dataset**: Upload your organized audio dataset to your Google Drive following the structure described above.
2.  **Configure Path**: Open the `gan_audio.py` script and modify the `BASE_PATH` variable in the main execution block (`if __name__ == '__main__':`) to point to your dataset's root directory on Google Drive.

    ```python
    # --- Paths and Data Setup ---
    BASE_PATH = 'drive/MyDrive/your_dataset_folder/' # <-- CHANGE THIS
    ```

3.  **Run in Google Colab**:
    * Upload the `gan_audio.py` file to your Colab environment or simply copy and paste its contents into a new Colab notebook.
    * Ensure your Colab runtime is set to use a **GPU accelerator** for faster training (`Runtime -> Change runtime type -> Hardware accelerator -> GPU`).
    * Run the cells sequentially. You will be prompted to authorize Colab to access your Google Drive.

4.  **Training**: The script will begin the training process. The progress for each epoch, including the Generator and Discriminator losses, will be displayed using `tqdm`.

---

## 📊 Expected Outputs

During and after training, the script will create two new folders in your Colab environment's root directory:

1.  `gan_spectrogram_plots/`: Contains PNG images of generated spectrograms for each category, saved at the end of each epoch (e.g., `epoch_001.png`, `epoch_002.png`, etc.). This helps visualize the generator's learning progress.
2.  `gan_generated_audio/`: Contains the generated `.wav` audio files for each category at the end of each epoch (e.g., `category_1_ep1.wav`, `category_2_ep1.wav`, etc.). You can listen to these files directly in the Colab output or download them.

---

## 📂 Code Structure

The `gan_audio.py` script is organized into five main sections for clarity:

1.  **`0. IMPORTS & INITIAL SETUP`**: Imports all necessary libraries and handles Google Drive mounting.
2.  **`1. DATASET CLASS`**: Defines the `TrainAudioSpectrogramDataset` class for loading, preprocessing, and labeling the audio data.
3.  **`2. GAN MODEL DEFINITIONS`**: Defines the `CGAN_Generator` and `CGAN_Discriminator` neural network architectures.
4.  **`3. UTILITY FUNCTIONS`**: Contains helper functions for generating audio from spectrograms and for saving/displaying the output.
5.  **`4. GAN TRAINING FUNCTION`**: The core `train_gan` function that orchestrates the training loop for the generator and discriminator.
6.  **`5. MAIN EXECUTION BLOCK`**: Sets configuration parameters (learning rate, batch size, etc.), initializes the dataset, models, and starts the training process.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
