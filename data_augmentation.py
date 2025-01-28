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
    os.makedirs(output_folder, exist_ok=True)

    # Iteracja po folderach i plikach w folderze wejściowym
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            input_path = os.path.join(root, filename)

            # Sprawdzenie, czy plik jest plikiem audio
            if filename.endswith(".wav"):
                try:
                    # Wczytanie pliku audio
                    waveform, sample_rate = torchaudio.load(input_path)

                    # Dodanie szumu do sygnału
                    noisy_waveform = add_noise(waveform, noise_level=noise_level)

                    # Tworzenie nowej nazwy pliku
                    relative_path = os.path.relpath(input_path, input_folder)
                    new_filename = os.path.splitext(relative_path)[0] + '_noise.wav'

                    # Ścieżka do zapisu zmodyfikowanego pliku
                    output_path = os.path.join(output_folder, new_filename)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    # Zapisanie przetworzonego pliku audio
                    torchaudio.save(output_path, noisy_waveform, sample_rate)
                    print(f"Przetworzono plik: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"Nie udało się przetworzyć pliku {input_path}: {e}")


def pitch_shift_folder(input_folder, output_folder, n_steps):
    os.makedirs(output_folder, exist_ok=True)

    # Iteracja po folderach i plikach w folderze wejściowym
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            input_path = os.path.join(root, filename)

            # Sprawdzenie, czy plik jest plikiem audio
            if filename.endswith(".wav"):
                try:
                    # Wczytanie pliku audio
                    #waveform, sample_rate = torchaudio.load(input_path,normalize=True)
                    y, sr = librosa.load(input_path, sr=None)


                    # Zastosowanie pitch shiftingu
                    #pitch_shift = T.PitchShift(sample_rate=sample_rate,n_steps=n_steps)
                    #shifted_waveform = pitch_shift(waveform)
                    y_shifted = librosa.effects.pitch_shift(y,sr=sr,n_steps=n_steps)
                    #shifted_waveform = shifted_waveform.detach()


                    # Tworzenie nowej nazwy pliku
                    relative_path = os.path.relpath(input_path, input_folder)
                    new_filename = os.path.splitext(relative_path)[0] + '_pitchshift.wav'

                    # Ścieżka do zapisu zmodyfikowanego pliku
                    output_path = os.path.join(output_folder, new_filename)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    # Zapisanie zmodyfikowanego pliku
                    #torchaudio.save(output_path, shifted_waveform, sample_rate)
                    sf.write(output_path, y_shifted, sr)

                except Exception as e:
                    print(f"Nie udało się przetworzyć pliku {input_path}: {e}")
        print(f'przetworzono folder: {root}')

# Ścieżki do folderów
input_folder = 'augmentation_data/train'
#input_folder = 'augmentation_test/input'
output_folder = 'augmentation_data/pitchshift'
#output_folder = 'augmentation_test/output'

# Przykład użycia funkcji (wybierz jedną z opcji)

# Dodanie szumu do wszystkich plików
#noise_level = 0.005
#add_noise_folder(input_folder, output_folder, noise_level)

# Zastosowanie pitch shiftingu do wszystkich plików
n_steps = 2
pitch_shift_folder(input_folder, output_folder, n_steps)
