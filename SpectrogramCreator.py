import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


class SpectrogramCreator:

    def create_spectrogram(self, file_path):
        y, sr = librosa.load(file_path, sr=None)  # return audio time series and sample rate
        D = librosa.stft(y,n_fft=4096, hop_length=256)  # Furier transformations
        S_db = librosa.amplitude_to_db(abs(D), ref=np.max)  # convert amplitude to db, abs because D has complex values
        return S_db, sr

    def plot_spectrogram(self, S_db, sr):
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
