import os
import torchaudio.transforms as T
import torchaudio
import librosa
import soundfile as sf
import torch


def add_noise(waveform, noise_level):
    noise = noise_level * torch.randn_like(waveform)
    noisy_waveform = waveform + noise
    noisy_waveform = torch.clamp(noisy_waveform, -1.0, 1.0)
    return noisy_waveform


def add_noise_folder(input_folder, output_folder, noise_level):
    """
    Iterates over subfolders and files in the input folder, and applies Gaussian noise to every .wav file.
    :param input_folder: Folder containing original .wav files
    :param output_folder: Folder to save processed .wav files
    :param noise_level: Float indicating the noise amplitude
    """
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over subfolders and files in the input folder
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            input_path = os.path.join(root, filename)

            # Check if the file is an audio file
            if filename.endswith(".wav"):
                try:
                    # Load the audio file
                    waveform, sample_rate = torchaudio.load(input_path)

                    # Add noise to the signal
                    noisy_waveform = add_noise(waveform, noise_level=noise_level)

                    # Generate a new filename
                    relative_path = os.path.relpath(input_path, input_folder)
                    new_filename = os.path.splitext(relative_path)[0] + '_noise.wav'

                    # Path to save the modified file
                    output_path = os.path.join(output_folder, new_filename)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    # Save the processed audio file
                    torchaudio.save(output_path, noisy_waveform, sample_rate)
                    print(f"Przetworzono plik: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"Nie udało się przetworzyć pliku {input_path}: {e}")


def pitch_shift_folder(input_folder, output_folder, n_steps):
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over subfolders and files in the input folder
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            input_path = os.path.join(root, filename)

            # Check if the file is an audio file
            if filename.endswith(".wav"):
                try:
                    # Load the audio file
                    #waveform, sample_rate = torchaudio.load(input_path,normalize=True)
                    y, sr = librosa.load(input_path, sr=None)


                    # Apply pitch shifting using librosa
                    #pitch_shift = T.PitchShift(sample_rate=sample_rate,n_steps=n_steps)
                    #shifted_waveform = pitch_shift(waveform)
                    y_shifted = librosa.effects.pitch_shift(y,sr=sr,n_steps=n_steps)
                    #shifted_waveform = shifted_waveform.detach()


                    # Generate a new filename
                    relative_path = os.path.relpath(input_path, input_folder)
                    new_filename = os.path.splitext(relative_path)[0] + '_pitchshift.wav'

                    # Path to save the modified file
                    output_path = os.path.join(output_folder, new_filename)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    # Save the modified file
                    #torchaudio.save(output_path, shifted_waveform, sample_rate)
                    sf.write(output_path, y_shifted, sr)

                except Exception as e:
                    print(f"Nie udało się przetworzyć pliku {input_path}: {e}")
        print(f'przetworzono folder: {root}')

# Paths to folders
input_folder = 'augmentation_data/train'
#input_folder = 'augmentation_test/input'
output_folder = 'augmentation_data/pitchshift'
#output_folder = 'augmentation_test/output'

# Example usage (choose one of the options below)

# Add noise to all files
#noise_level = 0.005
#add_noise_folder(input_folder, output_folder, noise_level)

# Apply pitch shifting to all files
n_steps = 2
pitch_shift_folder(input_folder, output_folder, n_steps)
